#!/usr/bin/env bash
# =============================================================================
# tts_api.sh — one-shot bootstrap + launcher for the unified vLLM-Omni
#              voice-cloning TTS API (Qwen3-TTS Base + OmniVoice).
#
# Run on a fresh pod (or after reboot):
#     bash tts_api.sh            # setup if needed, then start
#
# Subcommands:
#     bash tts_api.sh setup      # install deps + venvs + models only
#     bash tts_api.sh start      # start workers + gateway (refuses if already up)
#     bash tts_api.sh restart    # stop then start
#     bash tts_api.sh stop       # stop everything
#     bash tts_api.sh status     # health of gateway + workers
#     bash tts_api.sh test       # clone a voice + generate on both models
#     bash tts_api.sh docs       # print the API reference
#
# Everything self-contained: this script writes worker.py + gateway.py itself.
# =============================================================================
set -uo pipefail

# ---------- config -----------------------------------------------------------
ROOT=/root
REPO=$ROOT/vllm-omni
REPO_URL=${REPO_URL:-https://github.com/pavanyellow/vllm-omni.git}
APP=$ROOT/tts
LOGDIR=$APP/logs
CFGDIR=$ROOT/tts_configs
VOICES=$ROOT/tts_voices
VENV_MAIN=$ROOT/tts-venv          # transformers 5.9  -> OmniVoice worker + gateway
VENV_QWEN3=$ROOT/tts-venv-qwen3   # transformers 4.57 -> Qwen3 worker
PY_MAIN=$VENV_MAIN/bin/python
PY_QWEN3=$VENV_QWEN3/bin/python

GW_HOST=${TTS_GATEWAY_HOST:-0.0.0.0}
GW_PORT=${TTS_GATEWAY_PORT:-8000}
QWEN3_PORT=8101
OMNI_PORT=8102

VLLM_VERSION=0.21.0
TF_MAIN=5.9.0          # OmniVoice cloning needs >=5.3; vLLM 0.21 forbids 5.0-5.4
TF_QWEN3=4.57.6        # Qwen3 vendored decoder needs pre-rename create_causal_mask
QWEN3_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base
OMNI_MODEL=k2-fsa/OmniVoice
REF_URL=https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav

# caches on FAST local disk — this box defaults them to the slow /workspace net mount
export UV_CACHE_DIR=${UV_CACHE_DIR:-$ROOT/.cache/uv}
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-600}
export HF_HOME=${HF_HOME:-$ROOT/.hf}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$ROOT/.hf/hub}
export VLLM_OMNI_REPO=$REPO
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "$UV_CACHE_DIR" "$HF_HOME/hub" "$APP" "$LOGDIR" "$CFGDIR" "$VOICES"

log(){ echo "==> $*"; }
err(){ echo "!! $*" >&2; }
retry(){ local n=$1; shift; local i=1; until "$@"; do [ "$i" -ge "$n" ] && { err "giving up: $*"; return 1; }; err "retry $((++i))/$n"; sleep 4; done; }

# ---------- write the worker + gateway source --------------------------------
write_app_files(){
  log "writing $APP/worker.py and $APP/gateway.py"

  cat > "$APP/worker.py" <<'WORKER_EOF'
"""Single-model TTS worker wrapping the vLLM-Omni offline engine.

Each worker process loads ONE model (Qwen3-TTS Base or OmniVoice) and exposes a
tiny HTTP API. Both engines support voice cloning from a reference clip via the
offline path (the stock vllm-omni HTTP server only exposes OmniVoice "auto
voice", which is why we drive the offline engine directly here).
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
from fastapi.responses import Response
from pydantic import BaseModel

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

REPO = os.environ.get("VLLM_OMNI_REPO", "/root/vllm-omni")

QWEN3_BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
OMNIVOICE_MODEL = "k2-fsa/OmniVoice"

QWEN3_LANGS = {
    "Auto", "Chinese", "English", "Japanese", "Korean", "German",
    "French", "Russian", "Portuguese", "Spanish", "Italian",
}


class InferRequest(BaseModel):
    text: str
    ref_audio: str | None = None
    ref_text: str | None = None
    language: str | None = None
    instruct: str | None = None
    max_new_tokens: int = 2048


def _load_qwen3_prompt_estimator():
    path = os.path.join(REPO, "examples/offline_inference/text_to_speech/qwen3_tts/end2end.py")
    spec = importlib.util.spec_from_file_location("qwen3_e2e", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._estimate_prompt_len


def _extract_audio(request_output) -> tuple[np.ndarray, int]:
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

    from vllm_omni import Omni

    if engine_name == "qwen3":
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams  # noqa: F401
        estimate_prompt_len = _load_qwen3_prompt_estimator()
        cfg = config or os.path.join(REPO, "vllm_omni/deploy/qwen3_tts.yaml")
        t0 = time.time()
        omni = Omni(model=QWEN3_BASE_MODEL, stage_configs_path=cfg, log_stats=False)
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
        omni = Omni(model=OMNIVOICE_MODEL, stage_configs_path=cfg, log_stats=False)
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
                prompt, sampling_params_list=[OmniDiffusionSamplingParams()], use_tqdm=False)
            return _extract_audio(results[0].request_output)
    else:
        raise SystemExit(f"unknown engine {engine_name!r}")

    @app.get("/health")
    def health():
        return {"engine": engine_name, "model": state["model"],
                "ready": state["ready"] and state["omni"] is not None, "error": state["error"]}

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
            state["ready"] = False
            state["error"] = repr(e)
            raise HTTPException(500, f"generation failed: {e!r}")
        return Response(content=wav, media_type="audio/wav",
                        headers={"X-Sample-Rate": str(sr), "X-Duration-Sec": f"{dur:.2f}",
                                 "X-Gen-Sec": f"{dt:.2f}"})

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["qwen3", "omnivoice"])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--gpu-mem", type=float, default=0.4)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    app = build_app(args.engine, args.gpu_mem, args.config)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
WORKER_EOF

  cat > "$APP/gateway.py" <<'GATEWAY_EOF'
"""Unified TTS gateway: voice cloning + generation across Qwen3-TTS and OmniVoice.

  1. POST /voices  (upload a reference clip + transcript)  -> voice_id
  2. POST /tts     ({text, voice_id, model})               -> WAV bytes

A voice_id is just a stored reference clip, so the SAME id works with either model.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import httpx
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

VOICES_DIR = Path(os.environ.get("TTS_VOICES_DIR", "/root/tts_voices"))
VOICES_DIR.mkdir(parents=True, exist_ok=True)

WORKERS = {
    "qwen3": os.environ.get("QWEN3_WORKER_URL", "http://127.0.0.1:8101"),
    "omnivoice": os.environ.get("OMNIVOICE_WORKER_URL", "http://127.0.0.1:8102"),
}
DEFAULT_MODEL = "qwen3"
REQUEST_TIMEOUT = float(os.environ.get("TTS_REQUEST_TIMEOUT", "600"))

app = FastAPI(title="vllm-omni TTS gateway")


def _voice_dir(voice_id: str) -> Path:
    if not voice_id or "/" in voice_id or "\\" in voice_id or ".." in voice_id:
        raise HTTPException(400, "invalid voice_id")
    return VOICES_DIR / voice_id


def _read_meta(voice_id: str) -> dict:
    meta = _voice_dir(voice_id) / "meta.json"
    if not meta.exists():
        raise HTTPException(404, f"voice '{voice_id}' not found")
    return json.loads(meta.read_text())


def _list_voices() -> list[dict]:
    out = []
    for d in sorted(VOICES_DIR.iterdir()):
        m = d / "meta.json"
        if m.is_file():
            try:
                out.append(json.loads(m.read_text()))
            except Exception:
                pass
    return out


class TTSRequest(BaseModel):
    text: str
    voice_id: str
    model: str = DEFAULT_MODEL
    language: str | None = None
    instruct: str | None = None
    max_new_tokens: int = 2048


@app.get("/")
def root():
    return {
        "service": "vllm-omni unified TTS (voice-cloning)",
        "models": list(WORKERS),
        "default_model": DEFAULT_MODEL,
        "workflow": [
            "POST /voices (multipart audio_sample=@clip.wav, ref_text=..., name=...) -> {voice_id}",
            "POST /tts {text, voice_id, model} -> audio/wav",
        ],
        "endpoints": ["GET /health", "POST /voices", "GET /voices",
                      "GET /voices/{id}", "DELETE /voices/{id}", "POST /tts"],
    }


@app.get("/health")
async def health():
    workers = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in WORKERS.items():
            try:
                r = await client.get(f"{url}/health")
                workers[name] = r.json()
            except Exception as e:
                workers[name] = {"ready": False, "error": repr(e)}
    return {"gateway": "ok", "voices_registered": len(_list_voices()), "workers": workers}


@app.post("/voices")
async def create_voice(
    audio_sample: UploadFile = File(...),
    ref_text: str = Form(...),
    name: str = Form(""),
):
    raw = await audio_sample.read()
    if not raw:
        raise HTTPException(400, "empty upload")
    voice_id = uuid.uuid4().hex[:12]
    vdir = _voice_dir(voice_id)
    vdir.mkdir(parents=True, exist_ok=True)
    src = vdir / f"upload_{audio_sample.filename or 'clip'}"
    src.write_bytes(raw)
    clip = vdir / "clip.wav"
    try:
        data, sr = sf.read(str(src), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        sf.write(str(clip), mono, sr, format="WAV", subtype="PCM_16")
        duration = len(mono) / sr
    except Exception as e:
        raise HTTPException(400, f"could not decode audio: {e!r}")
    finally:
        src.unlink(missing_ok=True)
    meta = {
        "voice_id": voice_id, "name": name or voice_id, "ref_text": ref_text,
        "clip_path": str(clip), "sample_rate": int(sr),
        "duration_sec": round(float(duration), 2),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (vdir / "meta.json").write_text(json.dumps(meta, indent=2))
    if not (1.0 <= duration <= 40.0):
        meta["warning"] = f"reference is {duration:.1f}s; 3-30s clean speech clones best"
    return JSONResponse(meta)


@app.get("/voices")
def list_voices():
    return {"voices": _list_voices()}


@app.get("/voices/{voice_id}")
def get_voice(voice_id: str):
    return _read_meta(voice_id)


@app.delete("/voices/{voice_id}")
def delete_voice(voice_id: str):
    vdir = _voice_dir(voice_id)
    if not vdir.exists():
        raise HTTPException(404, f"voice '{voice_id}' not found")
    for p in vdir.iterdir():
        p.unlink(missing_ok=True)
    vdir.rmdir()
    return {"deleted": voice_id}


@app.post("/tts")
async def tts(req: TTSRequest):
    if req.model not in WORKERS:
        raise HTTPException(400, f"unknown model '{req.model}'; choose {list(WORKERS)}")
    if not req.text or not req.text.strip():
        raise HTTPException(400, "text is required")
    meta = _read_meta(req.voice_id)
    payload = {
        "text": req.text, "ref_audio": meta["clip_path"], "ref_text": meta.get("ref_text", ""),
        "language": req.language, "instruct": req.instruct, "max_new_tokens": req.max_new_tokens,
    }
    url = WORKERS[req.model]
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            r = await client.post(f"{url}/infer", json=payload)
    except Exception as e:
        raise HTTPException(502, f"worker '{req.model}' unreachable: {e!r}")
    if r.status_code != 200:
        raise HTTPException(r.status_code, f"worker '{req.model}': {r.text}")
    return Response(content=r.content, media_type="audio/wav",
                    headers={"X-Model": req.model, "X-Voice-Id": req.voice_id,
                             "X-Sample-Rate": r.headers.get("X-Sample-Rate", ""),
                             "X-Duration-Sec": r.headers.get("X-Duration-Sec", ""),
                             "X-Gen-Sec": r.headers.get("X-Gen-Sec", "")})


def main():
    host = os.environ.get("TTS_GATEWAY_HOST", "0.0.0.0")
    port = int(os.environ.get("TTS_GATEWAY_PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
GATEWAY_EOF
}

# ---------- setup ------------------------------------------------------------
ensure_system_deps(){
  log "system deps (espeak-ng, ffmpeg, libsndfile, git, curl)"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq >/dev/null 2>&1 || true
  apt-get install -y -qq espeak-ng ffmpeg libsndfile1 git curl >/dev/null 2>&1 || err "apt step non-fatal"
  command -v uv >/dev/null 2>&1 || { log "installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
}

ensure_repo(){
  if [ ! -d "$REPO/.git" ]; then
    log "cloning $REPO_URL"
    retry 3 git clone "$REPO_URL" "$REPO"
  else
    log "repo present at $REPO"
  fi
}

setup_main_venv(){
  if [ -f "$VENV_MAIN/.tts_ready" ]; then log "main venv ready (skip)"; return; fi
  log "building main venv (OmniVoice + gateway, transformers $TF_MAIN)"
  [ -x "$PY_MAIN" ] || uv venv "$VENV_MAIN" --python 3.12
  ( cd "$REPO"
    VIRTUAL_ENV="$VENV_MAIN" retry 4 uv pip install --python "$PY_MAIN" "vllm==$VLLM_VERSION"
    VIRTUAL_ENV="$VENV_MAIN" retry 3 uv pip install --python "$PY_MAIN" -e .
    VIRTUAL_ENV="$VENV_MAIN" retry 3 uv pip install --python "$PY_MAIN" \
        "transformers==$TF_MAIN" tokenizers torchaudio soundfile huggingface_hub \
        fastapi "uvicorn[standard]" python-multipart httpx pyyaml ) || { err "main venv build failed"; return 1; }
  touch "$VENV_MAIN/.tts_ready"
}

setup_qwen3_venv(){
  if [ -f "$VENV_QWEN3/.tts_ready" ]; then log "qwen3 venv ready (skip)"; return; fi
  log "building qwen3 venv (transformers $TF_QWEN3)"
  [ -x "$PY_QWEN3" ] || uv venv "$VENV_QWEN3" --python 3.12
  ( cd "$REPO"
    VIRTUAL_ENV="$VENV_QWEN3" retry 4 uv pip install --python "$PY_QWEN3" "vllm==$VLLM_VERSION"
    VIRTUAL_ENV="$VENV_QWEN3" retry 3 uv pip install --python "$PY_QWEN3" -e .
    VIRTUAL_ENV="$VENV_QWEN3" retry 3 uv pip install --python "$PY_QWEN3" \
        "transformers==$TF_QWEN3" soundfile torchaudio fastapi "uvicorn[standard]" httpx ) || { err "qwen3 venv build failed"; return 1; }
  touch "$VENV_QWEN3/.tts_ready"
}

download_models(){
  log "downloading models (cached after first run)"
  retry 4 "$VENV_MAIN/bin/hf" download "$QWEN3_MODEL" >/dev/null && log "  $QWEN3_MODEL ok"
  retry 4 "$VENV_MAIN/bin/hf" download "$OMNI_MODEL"  >/dev/null && log "  $OMNI_MODEL ok"
}

patch_omnivoice_config(){
  "$PY_MAIN" - "$REPO/vllm_omni/deploy/omnivoice.yaml" "$CFGDIR/omnivoice.yaml" <<'PY'
import sys, yaml
src, dst = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(src))
for st in cfg.get("stage_args", []):
    st.setdefault("engine_args", {})["gpu_memory_utilization"] = 0.3
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
PY
}

do_setup(){
  ensure_system_deps
  ensure_repo
  write_app_files
  setup_main_venv  || return 1
  setup_qwen3_venv || return 1
  download_models
  patch_omnivoice_config
  log "setup complete"
}

# ---------- run --------------------------------------------------------------
gateway_healthy(){ curl -sf "http://127.0.0.1:$GW_PORT/health" >/dev/null 2>&1; }

wait_ready(){  # name port
  local name=$1 port=$2
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$port/health" 2>/dev/null | grep -q '"ready": *true' && { log "$name ready"; return 0; }
    kill -0 "$(cat "$LOGDIR/$name.pid" 2>/dev/null)" 2>/dev/null || { err "$name died; see $LOGDIR/$name.log"; return 1; }
    sleep 5
  done
  err "$name not ready; see $LOGDIR/$name.log"; return 1
}

do_start(){
  if gateway_healthy; then log "gateway already running on :$GW_PORT (use 'restart')"; do_status; return 0; fi
  write_app_files
  patch_omnivoice_config
  cd "$REPO"

  log "starting qwen3 worker (:$QWEN3_PORT, transformers $TF_QWEN3)"
  PATH="$VENV_QWEN3/bin:$PATH" nohup "$PY_QWEN3" "$APP/worker.py" --engine qwen3 --port "$QWEN3_PORT" \
      > "$LOGDIR/qwen3.log" 2>&1 & echo $! > "$LOGDIR/qwen3.pid"

  log "starting omnivoice worker (:$OMNI_PORT, transformers $TF_MAIN)"
  PATH="$VENV_MAIN/bin:$PATH" nohup "$PY_MAIN" "$APP/worker.py" --engine omnivoice --port "$OMNI_PORT" \
      --config "$CFGDIR/omnivoice.yaml" > "$LOGDIR/omnivoice.log" 2>&1 & echo $! > "$LOGDIR/omnivoice.pid"

  wait_ready qwen3 "$QWEN3_PORT" || true
  wait_ready omnivoice "$OMNI_PORT" || true

  log "starting gateway (:$GW_PORT on $GW_HOST)"
  TTS_GATEWAY_HOST="$GW_HOST" TTS_GATEWAY_PORT="$GW_PORT" \
  PATH="$VENV_MAIN/bin:$PATH" nohup "$PY_MAIN" "$APP/gateway.py" > "$LOGDIR/gateway.log" 2>&1 & echo $! > "$LOGDIR/gateway.pid"
  sleep 3
  do_status
  echo
  log "API up. Port-forward from your laptop:  ssh -N -L $GW_PORT:localhost:$GW_PORT root@<pod-ip> -p <ssh-tcp-port>"
  log "then hit http://localhost:$GW_PORT  (run 'bash $0 docs' for the API reference)"
}

do_stop(){
  for name in gateway qwen3 omnivoice; do
    local pidf="$LOGDIR/$name.pid"
    if [ -f "$pidf" ]; then
      local pid; pid=$(cat "$pidf")
      kill -0 "$pid" 2>/dev/null && { log "stopping $name (pid $pid)"; kill "$pid" 2>/dev/null; pkill -P "$pid" 2>/dev/null; }
      rm -f "$pidf"
    fi
  done
  pkill -f "$APP/worker.py" 2>/dev/null
  pkill -f "$APP/gateway.py" 2>/dev/null
  log "stopped"
}

do_status(){
  if gateway_healthy; then
    curl -s "http://127.0.0.1:$GW_PORT/health" | "$PY_MAIN" -m json.tool 2>/dev/null || curl -s "http://127.0.0.1:$GW_PORT/health"
  else
    err "gateway not responding on :$GW_PORT"
    echo "workers: qwen3=$(curl -sf http://127.0.0.1:$QWEN3_PORT/health >/dev/null 2>&1 && echo up || echo down) omnivoice=$(curl -sf http://127.0.0.1:$OMNI_PORT/health >/dev/null 2>&1 && echo up || echo down)"
  fi
}

do_test(){
  gateway_healthy || { err "gateway not up; run 'bash $0 start' first"; return 1; }
  local ref=$ROOT/ref_clone.wav
  [ -s "$ref" ] || { log "fetching a sample reference clip"; curl -sL -o "$ref" "$REF_URL"; }
  local reftext="Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
  log "registering voice"
  local vid
  vid=$(curl -s -X POST "http://127.0.0.1:$GW_PORT/voices" \
        -F "audio_sample=@$ref" -F "ref_text=$reftext" -F "name=smoke_test" \
        | "$PY_MAIN" -c "import sys,json;print(json.load(sys.stdin)['voice_id'])")
  log "voice_id=$vid"
  for m in qwen3 omnivoice; do
    log "generating with $m"
    curl -s -X POST "http://127.0.0.1:$GW_PORT/tts" -H "Content-Type: application/json" \
      -d "{\"text\":\"Hello, this is the $m cloned voice.\",\"voice_id\":\"$vid\",\"model\":\"$m\",\"language\":\"English\"}" \
      -D "$LOGDIR/test_$m.hdr" --output "$ROOT/test_$m.wav"
    "$PY_MAIN" -c "import soundfile as sf,numpy as np;a,sr=sf.read('$ROOT/test_$m.wav');a=np.asarray(a).reshape(-1);print(f'  $m -> {len(a)/sr:.2f}s @ {sr}Hz peak={abs(a).max():.2f} -> $ROOT/test_$m.wav')" \
      || err "  $m FAILED (see $LOGDIR/test_$m.hdr / $LOGDIR/$m.log)"
  done
}

do_docs(){ awk '/^=== API REFERENCE ===$/{f=1;next} /^API_REFERENCE_BLOCK$/{f=0} f' "$0"; }

# ---------- dispatch ---------------------------------------------------------
case "${1:-default}" in
  setup)   do_setup ;;
  start)   do_start ;;
  restart) do_stop; sleep 2; do_start ;;
  stop)    do_stop ;;
  status)  do_status ;;
  test)    do_test ;;
  docs)    do_docs ;;
  default) do_setup && do_start ;;
  *) echo "usage: bash $0 {setup|start|restart|stop|status|test|docs}"; exit 2 ;;
esac
exit 0

# The block below is printed by `bash tts_api.sh docs`. Keep it last.
: <<'API_REFERENCE_BLOCK'
=== API REFERENCE ===
Unified vLLM-Omni voice-cloning TTS API
=======================================

Base URL (after SSH port-forward):  http://localhost:8000
Models:  "qwen3" (Qwen3-TTS-12Hz-1.7B-Base), "omnivoice" (k2-fsa/OmniVoice)
Workflow is cloning-first: register a reference clip -> voice_id -> synthesize.

-------------------------------------------------------------------------------
GET /health
  -> {"gateway":"ok","voices_registered":N,"workers":{"qwen3":{...,"ready":true},
      "omnivoice":{...,"ready":true}}}

-------------------------------------------------------------------------------
POST /voices            (multipart/form-data)   -- clone/register a voice
  fields:
    audio_sample  (file, required)  reference clip: wav/mp3/flac, 3-30s clean speech
    ref_text      (str,  required)  exact transcript of the clip
    name          (str,  optional)  friendly label
  -> 200 JSON:
    {"voice_id":"ab12cd34ef56","name":"my_voice","ref_text":"...",
     "sample_rate":24000,"duration_sec":8.08,"created":"...Z"}

  example:
    curl -X POST http://localhost:8000/voices \
      -F "audio_sample=@reference.wav" \
      -F "ref_text=exact transcript of the clip" \
      -F "name=my_voice"

-------------------------------------------------------------------------------
GET    /voices              -> {"voices":[ {meta}, ... ]}
GET    /voices/{voice_id}   -> {meta}
DELETE /voices/{voice_id}   -> {"deleted":"<voice_id>"}

-------------------------------------------------------------------------------
POST /tts               (application/json)   -- synthesize with a cloned voice
  body:
    text          (str,  required)  text to speak
    voice_id      (str,  required)  from POST /voices
    model         (str,  optional)  "qwen3" (default) | "omnivoice"
    language      (str,  optional)  qwen3: English|Chinese|Auto|Japanese|Korean|
                                    German|French|Russian|Portuguese|Spanish|Italian
                                    omnivoice: ISO-ish code, e.g. en|zh|fr
    instruct      (str,  optional)  omnivoice voice-design hint, e.g. "female, low pitch"
    max_new_tokens(int,  optional)  default 2048
  -> 200 audio/wav  (16-bit mono, 24 kHz)
     response headers: X-Model, X-Voice-Id, X-Sample-Rate, X-Duration-Sec, X-Gen-Sec

  examples:
    # Qwen3
    curl -X POST http://localhost:8000/tts -H "Content-Type: application/json" \
      -d '{"text":"Hello from my cloned voice.","voice_id":"ab12cd34ef56","model":"qwen3","language":"English"}' \
      --output out_qwen3.wav

    # OmniVoice (same voice_id)
    curl -X POST http://localhost:8000/tts -H "Content-Type: application/json" \
      -d '{"text":"Bonjour, ceci est ma voix clonee.","voice_id":"ab12cd34ef56","model":"omnivoice","language":"fr"}' \
      --output out_omni.wav

-------------------------------------------------------------------------------
Notes
  - No streaming: each /tts call returns a complete WAV.
  - Same voice_id works with either model (it's just the stored reference clip).
  - Latency (A100, warm): omnivoice ~2s; qwen3 ~16s (first qwen3 call after start
    is ~100s, a one-time JIT compile).
  - Requests are serialized per model (one generation at a time) — fine for
    personal use, not tuned for concurrency.
API_REFERENCE_BLOCK
