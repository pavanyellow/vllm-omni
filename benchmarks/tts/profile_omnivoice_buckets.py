# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bucketed generation-time profile for OmniVoice.

Three duration buckets (short / medium / long) covering ~5s to ~30s,
five prompts per bucket, one warmup, wall-clock timing of
``Omni.generate()`` per prompt, then per-bucket averages.

Usage:
    python benchmarks/tts/profile_omnivoice_buckets.py \\
        --model /workspace/.hf/omnivoice \\
        --stage-config vllm_omni/deploy/omnivoice.yaml \\
        --output-dir /tmp/omnivoice_bench
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from vllm_omni.engine.arg_utils import nullify_stage_engine_defaults
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# OmniVoice's RuleDurationEstimator (ref "Nice to meet you." = 25 frames) maps
# Latin text at roughly 16.3 chars/sec of output audio on this checkpoint
# (measured empirically — the ref calibration is faster than real speech).
# Targeting bucket midpoints 7.5s / 17.5s / 27.5s gives ~120 / ~285 / ~450 chars.
BUCKETS: dict[str, list[str]] = {
    "short_5_10s": [
        "Good morning everyone, please take a seat so we can begin today's briefing right on time and not run over.",
        "The technician will arrive between two and four this afternoon to inspect the panel and replace the filter.",
        "We are now boarding rows fifteen through thirty for the morning flight to San Francisco at gate twenty two.",
        "Please remember to silence your phones before the performance gets under way and to keep the aisles clear.",
        "I have reviewed the document carefully and only have a few small comments before we send it out to the client.",
    ],
    "medium_15_20s": [
        (
            "Good morning everyone, thanks for joining the call. We will start with a quick "
            "recap of last week, then walk through the quarterly results and the road map for "
            "next quarter, and finally take questions from the team before we wrap up "
            "around the hour mark so people can get to their next meeting on time."
        ),
        (
            "The library was completely silent except for the soft hum of the air "
            "conditioning and the occasional turn of a page as students prepared for the "
            "long week of final exams that would begin first thing on Monday morning, and "
            "the librarian moved quietly between the desks to refill the water pitcher."
        ),
        (
            "When you open the application for the first time, please make sure to grant "
            "microphone access, sign in with your company email, and then complete the short "
            "interactive tutorial so the assistant can calibrate itself to your voice and "
            "to the typical background noise of the room you are using it in every day."
        ),
        (
            "Speech synthesis quality depends not only on the acoustic model but also on the "
            "text front end, the duration model, and the neural vocoder used to convert "
            "discrete tokens or spectrogram frames into a continuous waveform at the end of "
            "the pipeline, which is then played back through ordinary headphones or speakers."
        ),
        (
            "Travel updates for the morning commute: trains on the red line are running about "
            "ten minutes behind schedule because of a signal problem near the downtown "
            "interchange. Please consider taking the express bus if you need to be on time, "
            "and check the transit app for live arrival information before leaving the house."
        ),
    ],
    "long_25_30s": [
        (
            "In the early hours of the morning, before the sun had fully cleared the horizon, "
            "the small fishing village began to wake up. Lights flickered on inside wooden "
            "houses along the shore, thin trails of smoke rose from the kitchen chimneys, "
            "and the gentle sound of waves carried through the narrow stone streets all the "
            "way down toward the quiet harbor and the boats that were tied along the wharf, "
            "where the older fishermen were already inspecting their nets in the cold air."
        ),
        (
            "Modern text to speech systems combine several components into a single pipeline. "
            "A text frontend normalizes input and handles abbreviations, a duration model "
            "decides how long each segment should be, an acoustic model predicts audio "
            "tokens or spectrograms from the text, and finally a neural vocoder reconstructs "
            "a high fidelity waveform that can be played back through ordinary speakers or "
            "headphones, ideally with very low latency and minimal artifacts in the output."
        ),
        (
            "Please listen carefully to the following safety instructions before we depart. "
            "In the unlikely event of cabin depressurization, oxygen masks will drop "
            "automatically from the panel directly above your seat. Pull the nearest mask "
            "firmly toward you, place it over your nose and mouth, breathe normally, and "
            "assist any children or other passengers seated next to you only after your "
            "own mask is secure and the oxygen is flowing as it should be at that moment."
        ),
        (
            "The recipe is simple but it does take some patience. Begin by warming a "
            "generous splash of olive oil in a heavy pan over medium heat, then add finely "
            "chopped onions and cook them gently until they are soft and translucent. Stir "
            "in the garlic, let it become fragrant for about half a minute, and only then "
            "add the tomato paste, a small spoon of sugar, a generous pinch of salt and "
            "pepper, and finally the chopped tomatoes that have been drained and rinsed."
        ),
        (
            "Once upon a time there was a small clockmaker who lived alone at the very top "
            "of a windy hill. Every evening he would walk down to the village square, wind "
            "the great brass clock that stood beside the fountain, and exchange a few quiet "
            "words with the baker who was just closing up for the night, before climbing "
            "back up the long stone path to his workshop, his cat, and his tiny tools, and "
            "settling down to repair another broken watch that had been left at his door."
        ),
    ],
}


@dataclass
class Result:
    bucket: str
    prompt_idx: int
    gen_seconds: float
    audio_seconds: float
    out_path: str

    @property
    def rtf(self) -> float:
        # Real-time factor: < 1.0 means faster than real time.
        return self.gen_seconds / self.audio_seconds if self.audio_seconds > 0 else float("nan")


def _generate_once(omni: Omni, text: str) -> tuple[float, np.ndarray, int]:
    """Run one Omni.generate() call and return (wall_seconds, audio_np, sr)."""
    prompts = {"prompt": text}
    sampling_params = [OmniDiffusionSamplingParams()]

    t0 = time.perf_counter()
    outputs = list(omni.generate(prompts, sampling_params_list=sampling_params))
    elapsed = time.perf_counter() - t0

    if not outputs:
        raise RuntimeError("Empty output from Omni.generate()")
    ro = outputs[0].request_output
    mm = getattr(ro, "multimodal_output", None) or getattr(ro.outputs[0], "multimodal_output", None)
    if not mm or "audio" not in mm:
        raise RuntimeError("Output had no audio payload")
    audio = mm["audio"]
    sr = int(mm.get("sr", 24000))
    if not isinstance(audio, np.ndarray):
        audio = audio.cpu().numpy().squeeze()
    return elapsed, audio, sr


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/workspace/.hf/omnivoice")
    p.add_argument("--stage-config", default="vllm_omni/deploy/omnivoice.yaml")
    p.add_argument("--output-dir", default="/tmp/omnivoice_bench")
    p.add_argument(
        "--skip-save",
        action="store_true",
        help="Don't write WAVs to disk (faster, less disk).",
    )
    nullify_stage_engine_defaults(p)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[init] model={args.model}")
    omni = Omni(model=args.model, stage_configs_path=args.stage_config, log_stats=False)

    try:
        # Warmup with one short prompt so the first real measurement is not
        # dominated by lazy CUDA allocator / kernel autotune.
        print("[warmup] running one untimed generation")
        _generate_once(omni, "Warmup pass.")

        results: list[Result] = []
        for bucket, prompts in BUCKETS.items():
            print(f"\n[bucket] {bucket}")
            for i, text in enumerate(prompts):
                elapsed, audio, sr = _generate_once(omni, text)
                audio_secs = len(audio) / sr
                out_path = ""
                if not args.skip_save:
                    out_path = os.path.join(args.output_dir, f"{bucket}_{i:02d}.wav")
                    sf.write(out_path, audio, sr)
                res = Result(
                    bucket=bucket,
                    prompt_idx=i,
                    gen_seconds=elapsed,
                    audio_seconds=audio_secs,
                    out_path=out_path,
                )
                results.append(res)
                print(
                    f"  [{i}] gen={elapsed:6.3f}s  audio={audio_secs:5.2f}s"
                    f"  rtf={res.rtf:.3f}  chars={len(text)}"
                )

        # Per-bucket summary
        print("\n=== summary ===")
        print(f"{'bucket':<14} {'n':>2} {'avg_gen_s':>10} {'avg_audio_s':>12} {'avg_rtf':>9}")
        for bucket in BUCKETS:
            rows = [r for r in results if r.bucket == bucket]
            n = len(rows)
            avg_gen = sum(r.gen_seconds for r in rows) / n
            avg_audio = sum(r.audio_seconds for r in rows) / n
            avg_rtf = sum(r.rtf for r in rows) / n
            print(f"{bucket:<14} {n:>2} {avg_gen:>10.3f} {avg_audio:>12.3f} {avg_rtf:>9.3f}")

    finally:
        omni.close()
        print("\n[done]")


if __name__ == "__main__":
    main()
