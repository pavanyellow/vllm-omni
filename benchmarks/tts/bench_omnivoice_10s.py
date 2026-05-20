# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed 10-second OmniVoice latency benchmark.

A small, reproducible workload for tracking generation-time changes as
the OmniVoice pipeline is optimized. Generates ~10 s of audio for each of
five distinct prompts, repeated ``--reps`` times. Reports per-prompt and
overall medians with spread.

Run:
    .venv/bin/python benchmarks/tts/bench_omnivoice_10s.py
    .venv/bin/python benchmarks/tts/bench_omnivoice_10s.py --reps 5 --json out.json

The five prompts are tuned to land near 10 s of audio output each (the
rule-based duration estimator maps ~165 Latin chars → ~10 s on this
checkpoint). Vary the wording, not the length, when adapting them.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np

from vllm_omni.engine.arg_utils import nullify_stage_engine_defaults
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# Five prompts, each ~165 Latin chars → ~10 s of audio output.
PROMPTS_10S: list[str] = [
    (
        "Good morning everyone, thanks for joining the call today. We will start with a "
        "quick recap of last week and then walk through the latest results before we wrap up."
    ),
    (
        "The library was completely silent except for the soft hum of the air conditioning "
        "and the occasional turn of a page as students prepared for the long final exam week."
    ),
    (
        "When you open the application for the first time please grant microphone access, "
        "sign in with your company email, and complete the short tutorial that follows it."
    ),
    (
        "Speech synthesis quality depends not only on the acoustic model but also on the text "
        "frontend, the duration model, and the neural vocoder used at the end of the pipeline."
    ),
    (
        "Travel updates for the morning commute: trains on the red line are running about ten "
        "minutes late because of a signal problem near the downtown interchange this morning."
    ),
]
TARGET_AUDIO_SECONDS = 10.0
WARMUP_CALLS = 2


@dataclass
class CallResult:
    prompt_idx: int
    rep: int
    gen_seconds: float
    audio_seconds: float
    text_chars: int

    @property
    def rtf(self) -> float:
        return self.gen_seconds / self.audio_seconds if self.audio_seconds > 0 else float("nan")


@dataclass
class Summary:
    label: str
    n: int
    mean_ms: float
    median_ms: float
    std_ms: float
    p05_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    mean_audio_s: float
    mean_rtf: float


def _generate_once(omni: Omni, text: str) -> tuple[float, float]:
    """Returns (wall_seconds, audio_seconds). Timer brackets *only* generate()."""
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
    return elapsed, len(audio) / sr


def _summary(label: str, values: list[CallResult]) -> Summary:
    gen_ms = [r.gen_seconds * 1000 for r in values]
    audio_s = [r.audio_seconds for r in values]
    rtfs = [r.rtf for r in values]
    return Summary(
        label=label,
        n=len(values),
        mean_ms=statistics.mean(gen_ms),
        median_ms=statistics.median(gen_ms),
        std_ms=statistics.stdev(gen_ms) if len(gen_ms) > 1 else 0.0,
        p05_ms=float(np.percentile(gen_ms, 5)),
        p95_ms=float(np.percentile(gen_ms, 95)),
        min_ms=min(gen_ms),
        max_ms=max(gen_ms),
        mean_audio_s=statistics.mean(audio_s),
        mean_rtf=statistics.mean(rtfs),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/workspace/.hf/omnivoice")
    p.add_argument("--stage-config", default="vllm_omni/deploy/omnivoice.yaml")
    p.add_argument("--reps", type=int, default=3, help="Repetitions per prompt (default: 3 → 15 total).")
    p.add_argument(
        "--json",
        dest="json_out",
        default=None,
        help="Optional JSON file to dump full results for diffing across runs.",
    )
    p.add_argument(
        "--tag",
        default=None,
        help="Free-form label written into the JSON output (e.g. 'baseline', 'fp16').",
    )
    nullify_stage_engine_defaults(p)
    args = p.parse_args()

    print(f"[init] model={args.model}")
    t_init = time.perf_counter()
    omni = Omni(model=args.model, stage_configs_path=args.stage_config, log_stats=False)
    init_seconds = time.perf_counter() - t_init
    print(f"[init] engine ready in {init_seconds:.2f} s (one-time)")

    results: list[CallResult] = []
    try:
        # Warmup. The first call hits cold CUDA kernels; a second call on a
        # different shape clears most remaining JIT/autotune cost.
        print(f"\n[warmup] {WARMUP_CALLS} untimed calls")
        for i, text in enumerate(PROMPTS_10S[:WARMUP_CALLS]):
            _generate_once(omni, text)

        # Main: interleave prompts (P0,P1,...,P4,P0,P1,...) so any slow drift
        # (thermal throttling, allocator state) hits all prompts equally.
        print(f"\n[run] {len(PROMPTS_10S)} prompts × {args.reps} reps")
        for rep in range(args.reps):
            for idx, text in enumerate(PROMPTS_10S):
                gen_s, audio_s = _generate_once(omni, text)
                results.append(
                    CallResult(
                        prompt_idx=idx,
                        rep=rep,
                        gen_seconds=gen_s,
                        audio_seconds=audio_s,
                        text_chars=len(text),
                    )
                )
                print(
                    f"  rep={rep} p{idx}: gen={gen_s*1000:7.1f} ms"
                    f"  audio={audio_s:5.2f}s  rtf={gen_s/audio_s:.3f}"
                )

        # Per-prompt summary
        print("\n=== per-prompt (median across reps) ===")
        print(f"{'prompt':>6} {'n':>3} {'median_ms':>10} {'std_ms':>8} {'audio_s':>8} {'rtf':>6}")
        per_prompt: list[Summary] = []
        for idx in range(len(PROMPTS_10S)):
            rows = [r for r in results if r.prompt_idx == idx]
            s = _summary(f"prompt_{idx}", rows)
            per_prompt.append(s)
            print(f"  p{idx:<3} {s.n:>3} {s.median_ms:>10.1f} {s.std_ms:>8.1f} {s.mean_audio_s:>8.2f} {s.mean_rtf:>6.3f}")

        # Overall
        overall = _summary("overall", results)
        print("\n=== overall ===")
        print(f"  n           : {overall.n}")
        print(f"  mean        : {overall.mean_ms:.1f} ms")
        print(f"  median      : {overall.median_ms:.1f} ms")
        print(f"  std         : {overall.std_ms:.1f} ms")
        print(f"  p05 / p95   : {overall.p05_ms:.1f} / {overall.p95_ms:.1f} ms")
        print(f"  min / max   : {overall.min_ms:.1f} / {overall.max_ms:.1f} ms")
        print(f"  mean audio  : {overall.mean_audio_s:.2f} s   (target {TARGET_AUDIO_SECONDS}s)")
        print(f"  mean RTF    : {overall.mean_rtf:.3f}")

        if args.json_out:
            payload = {
                "tag": args.tag or "untagged",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_path": args.model,
                "reps": args.reps,
                "target_audio_seconds": TARGET_AUDIO_SECONDS,
                "init_seconds": init_seconds,
                "overall": asdict(overall),
                "per_prompt": [asdict(s) for s in per_prompt],
                "calls": [asdict(r) for r in results],
            }
            with open(args.json_out, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"\n[json] wrote {args.json_out}")

    finally:
        omni.close()
        print("\n[done]")


if __name__ == "__main__":
    main()
