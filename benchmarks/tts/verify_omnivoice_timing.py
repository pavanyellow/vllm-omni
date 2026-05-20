# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verification harness for OmniVoice timing measurements.

Sanity-checks the per-call wall-clock numbers reported by
``profile_omnivoice_buckets.py`` by:

  1. Reporting model-load time vs. per-call time (proving the engine is
     warm and not paying init cost per request).
  2. Running the same prompt N times to measure pure timing noise.
  3. Sweeping prompts across ~3-30s of output to fit a line and
     report residuals, R^2, and the implied per-frame cost.

Run it with the same venv as the profiling script:

    .venv/bin/python benchmarks/tts/verify_omnivoice_timing.py
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from vllm_omni.engine.arg_utils import nullify_stage_engine_defaults
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# Same prompt used many times → measures pure noise (CUDA/IPC/scheduler jitter).
NOISE_PROMPT = (
    "Speech synthesis quality depends not only on the acoustic model but also "
    "on the text frontend, the duration model, and the neural vocoder used to "
    "convert discrete tokens into a continuous audio waveform at the end."
)
NOISE_REPEATS = 10

# Prompts of varying length to make a fine-grained linear-fit test.
# Roughly 3 s up to 30 s in ~3 s steps.
SWEEP_PROMPTS = [
    "Hello world.",
    "Hello world, this is a quick test of the speech system today.",
    "Hello world, this is a quick test of the speech system today, please listen carefully to the output.",
    "Good morning everyone, please take a seat so we can begin the briefing on time and not run over the schedule.",
    "Speech synthesis quality depends not only on the acoustic model but also on the text frontend, the duration model, and the neural vocoder used at the end of the pipeline to render audio.",
    (
        "Good morning everyone, thanks for joining the call today. We will start with a quick recap of last "
        "week, then walk through the quarterly results and the road map for next quarter, and wrap up "
        "before the hour mark."
    ),
    (
        "In the early hours of the morning, before the sun had fully cleared the horizon, the small fishing "
        "village began to wake up. Lights flickered on inside the wooden houses along the shore and thin "
        "trails of smoke rose from the kitchen chimneys above the narrow stone streets."
    ),
    (
        "Modern text to speech systems combine several components into a single pipeline. A text frontend "
        "normalizes input and handles abbreviations, a duration model decides how long each segment should "
        "be, an acoustic model predicts audio tokens, and a neural vocoder reconstructs a high fidelity "
        "waveform at the end of the pipeline for playback through ordinary speakers or headphones."
    ),
    (
        "Please listen carefully to the following safety instructions before we depart on this evening's "
        "flight. In the unlikely event of cabin depressurization, oxygen masks will drop automatically from "
        "the panel directly above your seat. Pull the nearest mask firmly toward you, place it over your "
        "nose and mouth, breathe normally, and assist any children seated next to you only once your own "
        "mask is on and the oxygen is flowing as it should be at that moment in the cabin."
    ),
    (
        "Once upon a time there was a small clockmaker who lived alone at the very top of a windy hill. "
        "Every evening he would walk down to the village square, wind the great brass clock that stood "
        "beside the fountain, and exchange a few quiet words with the baker who was just closing up for "
        "the night, before climbing back up the long stone path to his workshop, his cat, and his tiny "
        "tools, and settling down to repair another broken watch that had been left at his door earlier."
    ),
]


def _generate_once(omni: Omni, text: str) -> tuple[float, int, int]:
    """Returns (wall_seconds, num_samples, sr). Timer brackets *only* generate()."""
    prompts = {"prompt": text}
    sampling_params = [OmniDiffusionSamplingParams()]

    t0 = time.perf_counter()
    outputs = list(omni.generate(prompts, sampling_params_list=sampling_params))
    elapsed = time.perf_counter() - t0

    ro = outputs[0].request_output
    mm = getattr(ro, "multimodal_output", None) or getattr(ro.outputs[0], "multimodal_output", None)
    audio = mm["audio"]
    sr = int(mm.get("sr", 24000))
    if not isinstance(audio, np.ndarray):
        audio = audio.cpu().numpy().squeeze()
    return elapsed, len(audio), sr


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/workspace/.hf/omnivoice")
    p.add_argument("--stage-config", default="vllm_omni/deploy/omnivoice.yaml")
    nullify_stage_engine_defaults(p)
    args = p.parse_args()

    # --- Phase 0: init (untimed in the per-call sense, but reported) ---
    print("[init] constructing Omni engine ...")
    t_init = time.perf_counter()
    omni = Omni(model=args.model, stage_configs_path=args.stage_config, log_stats=False)
    init_seconds = time.perf_counter() - t_init
    print(f"[init] engine construction took {init_seconds:.2f} s (one-time, not in per-call numbers)")

    try:
        # --- Phase 1: warmup, untimed ---
        print("\n[warmup] one untimed call to clear lazy CUDA init / kernel autotune")
        _generate_once(omni, "Warmup pass.")

        # --- Phase 2: noise floor on a single prompt ---
        print(f"\n[noise] running same prompt {NOISE_REPEATS}x to measure jitter")
        times = []
        n_samples_first = None
        sr_first = None
        for i in range(NOISE_REPEATS):
            t, n_samples, sr = _generate_once(omni, NOISE_PROMPT)
            times.append(t)
            if n_samples_first is None:
                n_samples_first, sr_first = n_samples, sr
            elif n_samples != n_samples_first:
                print(
                    f"  [warn] sample count drifted: call {i} got {n_samples} samples, "
                    f"first got {n_samples_first} (would indicate non-determinism)"
                )
            print(f"  call {i:2d}: {t*1000:7.1f} ms  ({n_samples} samples @ {sr} Hz)")
        arr = np.array(times)
        print(
            f"  → mean={arr.mean()*1000:.1f} ms  median={np.median(arr)*1000:.1f} ms  "
            f"std={arr.std()*1000:.1f} ms  min={arr.min()*1000:.1f}  max={arr.max()*1000:.1f}  "
            f"max-min={(arr.max()-arr.min())*1000:.1f} ms"
        )
        print(f"  audio = {n_samples_first/sr_first:.2f} s  (target_frames = {n_samples_first // 960})")

        # --- Phase 3: linear-fit sweep ---
        print(f"\n[sweep] {len(SWEEP_PROMPTS)} prompts across the full audio-duration range")
        sweep = []
        for text in SWEEP_PROMPTS:
            t, n_samples, sr = _generate_once(omni, text)
            target_frames = n_samples // 960
            audio_seconds = n_samples / sr
            sweep.append((target_frames, audio_seconds, t))
            print(
                f"  chars={len(text):>4}  frames={target_frames:>4}  "
                f"audio={audio_seconds:5.2f}s  gen={t*1000:7.1f} ms"
            )

        # Fit line: gen_time = a + b * frames
        frames = np.array([row[0] for row in sweep], dtype=float)
        gen = np.array([row[2] for row in sweep], dtype=float)
        b, a = np.polyfit(frames, gen, 1)
        pred = a + b * frames
        resid = gen - pred
        ss_res = np.sum(resid**2)
        ss_tot = np.sum((gen - gen.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        rmse = np.sqrt(ss_res / len(gen))

        # Also try quadratic to see if there's any super-linear curvature.
        c2, b2, a2 = np.polyfit(frames, gen, 2)

        print(f"\n[fit] linear:    gen = {a*1000:.1f} ms + {b*1000:.3f} ms × frames")
        print(f"      → equivalent: {b*1000*25:.1f} ms per second of audio + {a*1000:.0f} ms fixed")
        print(f"      → R² = {r2:.6f}   RMSE = {rmse*1000:.2f} ms")
        print(f"      quadratic: c={c2:.3e}  (>0 → super-linear; <0 → sub-linear)")
        print(f"\n[fit] residuals from linear (ms):")
        for (f_, _, _), r in zip(sweep, resid):
            print(f"      frames={f_:>4}   resid={r*1000:+6.1f} ms")

    finally:
        omni.close()
        print("\n[done]")


if __name__ == "__main__":
    main()
