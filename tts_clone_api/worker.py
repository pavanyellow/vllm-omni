"""Single-model TTS worker wrapping the vLLM-Omni offline engine.

Each worker process loads ONE model (Qwen3-TTS Base or OmniVoice) and exposes a
tiny HTTP API. Both engines support voice cloning from a reference clip via the
offline path (the stock vllm-omni HTTP server only exposes OmniVoice "auto
voice", which is why we drive the offline engine directly here).

Run:
    python worker.py --engine qwen3     --port 8101 --gpu-mem 0.45
    python worker.py --engine omnivoice --port 8102 --gpu-mem 0.30
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import os
import threading
import time

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

# spawn is required for vllm-omni mp executor stages.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

REPO = os.environ.get("VLLM_OMNI_REPO", "/root/vllm-omni")

QWEN3_BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
OMNIVOICE_MODEL = "k2-fsa/OmniVoice"

# Qwen3-TTS accepts a fixed language vocabulary; OmniVoice accepts ISO-ish codes.
QWEN3_LANGS = {
    "Auto", "Chinese", "English", "Japanese", "Korean", "German",
    "French", "Russian", "Portuguese", "Spanish", "Italian",
}


class InferRequest(BaseModel):
    text: str
    ref_audio: str | None = None  # local path or URL (cloning reference)
    ref_text: str | None = None  # transcript of the reference clip
    language: str | None = None
    instruct: str | None = None  # OmniVoice voice-design hint (optional)
    max_new_tokens: int = 2048


def _load_qwen3_prompt_estimator():
    """Import _estimate_prompt_len from the upstream offline example.

    Keeps the (intricate) prompt-length math in sync with the repo instead of
    forking it here.
    """
    path = os.path.join(
        REPO, "examples/offline_inference/text_to_speech/qwen3_tts/end2end.py"
    )
    spec = importlib.util.spec_from_file_location("qwen3_e2e", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._estimate_prompt_len


def _extract_audio(request_output) -> tuple[np.ndarray, int]:
    """Pull (mono float32 waveform, sample_rate) out of an Omni request output.

    Handles both shapes seen across models: top-level multimodal_output
    (OmniVoice) and per-output multimodal_output (Qwen3), audio as ndarray, a
    single tensor, or a list of chunk tensors.
    """
    mm = getattr(request_output, "multimodal_output", None)
    if not mm and getattr(request_output, "outputs", None):
        mm = getattr(request_output.outputs[0], "multimodal_output", None)
    if not mm or "audio" not in mm:
        raise RuntimeError("engine returned no audio in multimodal_output")

    audio = mm["audio"]
    if isinstance(audio, list):
        parts = [a if torch.is_tensor(a) else torch.as_tensor(a) for a in audio]
        audio = torch.cat(parts, dim=-1)
    if torch.is_tensor(audio):
        arr = audio.float().cpu().numpy()
    else:
        arr = np.asarray(audio, dtype=np.float32)
    arr = arr.squeeze().reshape(-1)

    sr = mm.get("sr", 24000)
    if isinstance(sr, (list, tuple)):
        sr = sr[-1]
    sr = int(sr.item() if hasattr(sr, "item") else sr)
    return arr, sr


def build_app(engine_name: str, gpu_mem: float, config: str | None) -> FastAPI:
    app = FastAPI(title=f"tts-worker:{engine_name}")
    lock = threading.Lock()
    state: dict = {"omni": None, "ready": False, "error": None, "model": None}

    # ---- model load (blocking; done before serving) --------------------------
    from vllm_omni import Omni

    if engine_name == "qwen3":
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams  # noqa: F401

        estimate_prompt_len = _load_qwen3_prompt_estimator()
        cfg = config or os.path.join(REPO, "vllm_omni/deploy/qwen3_tts.yaml")
        t0 = time.time()
        omni = Omni(
            model=QWEN3_BASE_MODEL,
            stage_configs_path=cfg,
            log_stats=False,
        )
        state.update(omni=omni, model=QWEN3_BASE_MODEL, ready=True)
        print(f"[worker:qwen3] loaded in {time.time() - t0:.1f}s", flush=True)

        def _infer(req: InferRequest) -> tuple[np.ndarray, int]:
            if not req.ref_audio:
                raise HTTPException(400, "qwen3 cloning requires ref_audio")
            language = req.language or "Auto"
            if language not in QWEN3_LANGS:
                language = "Auto"
            info = {
                "task_type": ["Base"],
                "ref_audio": [req.ref_audio],
                "ref_text": [req.ref_text or ""],
                "text": [req.text],
                "language": [language],
                "x_vector_only_mode": [False],
                "max_new_tokens": [int(req.max_new_tokens)],
            }
            inputs = {
                "prompt_token_ids": [0] * estimate_prompt_len(info, QWEN3_BASE_MODEL),
                "additional_information": info,
            }
            results = state["omni"].generate([inputs], use_tqdm=False)
            return _extract_audio(results[0].request_output)

    elif engine_name == "omnivoice":
        from vllm.multimodal.media.audio import load_audio
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        cfg = config or os.path.join(REPO, "vllm_omni/deploy/omnivoice.yaml")
        t0 = time.time()
        omni = Omni(
            model=OMNIVOICE_MODEL,
            stage_configs_path=cfg,
            log_stats=False,
        )
        state.update(omni=omni, model=OMNIVOICE_MODEL, ready=True)
        print(f"[worker:omnivoice] loaded in {time.time() - t0:.1f}s", flush=True)

        def _infer(req: InferRequest) -> tuple[np.ndarray, int]:
            mm_data: dict = {}
            mm_kwargs: dict = {}
            if req.ref_audio:
                audio_signal, sr = load_audio(req.ref_audio, sr=None)
                mm_data["audio"] = (audio_signal.astype(np.float32), sr)
                mm_kwargs["ref_text"] = req.ref_text or ""
                mm_kwargs["sample_rate"] = sr
            if req.language:
                mm_kwargs["lang"] = req.language
            if req.instruct:
                mm_kwargs["instruct"] = req.instruct
            prompt: dict = {"prompt": req.text}
            if mm_data:
                prompt["multi_modal_data"] = mm_data
            if mm_kwargs:
                prompt["mm_processor_kwargs"] = mm_kwargs
            results = state["omni"].generate(
                prompt,
                sampling_params_list=[OmniDiffusionSamplingParams()],
                use_tqdm=False,
            )
            return _extract_audio(results[0].request_output)

    else:
        raise SystemExit(f"unknown engine {engine_name!r}")

    # ---- routes --------------------------------------------------------------
    @app.get("/health")
    def health():
        return {
            "engine": engine_name,
            "model": state["model"],
            "ready": state["ready"] and state["omni"] is not None,
            "error": state["error"],
        }

    @app.post("/infer")
    async def infer(req: InferRequest):
        if not state["ready"] or state["omni"] is None:
            raise HTTPException(503, f"engine dead: {state['error']}")
        if not req.text or not req.text.strip():
            raise HTTPException(400, "text is required")

        def _run():
            with lock:
                t0 = time.time()
                arr, sr = _infer(req)
                dt = time.time() - t0
                buf = io.BytesIO()
                sf.write(buf, arr, sr, format="WAV", subtype="PCM_16")
                return buf.getvalue(), sr, len(arr) / sr, dt

        try:
            wav, sr, dur, dt = await run_in_threadpool(_run)
        except HTTPException:
            raise
        except Exception as e:
            # Omni.generate() closes the engine on failure -> mark dead.
            state["ready"] = False
            state["error"] = repr(e)
            raise HTTPException(500, f"generation failed: {e!r}")
        return Response(
            content=wav,
            media_type="audio/wav",
            headers={
                "X-Sample-Rate": str(sr),
                "X-Duration-Sec": f"{dur:.2f}",
                "X-Gen-Sec": f"{dt:.2f}",
            },
        )

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["qwen3", "omnivoice"])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--gpu-mem", type=float, default=0.4)
    ap.add_argument("--config", default=None, help="override deploy/stage config yaml")
    args = ap.parse_args()
    app = build_app(args.engine, args.gpu_mem, args.config)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
