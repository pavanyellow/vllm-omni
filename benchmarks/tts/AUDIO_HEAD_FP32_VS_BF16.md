# OmniVoice audio_head: FP32 vs BF16 (perf finding)

The `audio_head` stage of the OmniVoice generator is a single `nn.Linear(1024 → 8·1025)`
matmul that runs once per unmasking step (32× per request). Walking the roofline
from theory → measurement → kernel symbol identified that the deploy config is
forcing this matmul (and the rest of the model) into FP32 instead of BF16.

## Roofline vs measurement (canonical 10 s shape, B=2, S=280)

|                                | per call    | × 32 calls |
| ------------------------------ | ----------- | ---------- |
| theoretical BF16 roofline      | 9.5 µs      | 0.30 ms    |
| **observed FP32 (production)** | 273.04 µs   | **8.74 ms** |
| **observed BF16 (production)** | 32.98 µs    | **1.06 ms** |

FP32 observation is **29×** over the BF16 roofline.
BF16 observation is **3.5×** over the BF16 roofline (small-shape overhead).
FP32 → BF16 speedup in this stage: **8.3×** (273 / 33 µs per call).

## How the dtype was identified

Three independent confirmations agreed:

1. **Roofline ratio.** 273 µs / 9.5 µs = 29× over BF16 roofline (impossible
   efficiency), but 273 / 140 = 1.94× over FP32 roofline (consistent with
   ~55% of peak — normal for a small-M matmul).
2. **Kernel symbol.** `torch.profiler` reports the backing cuBLAS kernel as
   `sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize64x64x8_…` — the
   `f32f32_f32f32_f32` slug literally encodes (A, B, accumulator, C, output) =
   all FP32. Switching the model to BF16 picks a completely different kernel,
   `nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN` — a Hopper-native WGMMA path.
3. **Achieved TFLOPS.** Measured 36.8 TFLOPS = 55% of FP32 peak (67) but only
   3.7% of BF16 peak (989). No healthy BF16 kernel runs that far below peak.

## Source of the dtype choice

`vllm_omni/deploy/omnivoice.yaml`:

```yaml
engine_args:
  model_class_name: "OmniVoicePipeline"
  enforce_eager: true
  trust_remote_code: true
  distributed_executor_backend: "mp"
  dtype: "float32"     # ← forces every matmul, including audio_head, to FP32
```

## Scripts in this folder used to produce the numbers

- `bench_omnivoice_10s.py`        — end-to-end 10 s latency baseline (5 prompts × 3 reps)
- `profile_omnivoice_buckets.py`  — 5 s / 17 s / 27 s duration buckets
- `verify_omnivoice_timing.py`    — measurement-methodology checks (noise floor, linearity)
- `waterfall_avg.py`              — per-stage cumulative breakdown of a 10 s request
                                    (embed / attention / MLP / norms / audio_head / sampling / decoder)
