# OmniVoice BF16 generator waterfall (BF16 generator + FP32 decoder)

End-to-end per-stage breakdown when the OmniVoice generator runs in BF16 while
the decoder is kept in FP32. Both runs use the same script methodology
(per-block CUDA-event wraps, 20 samples across 5 prompts × 4 reps,
canonical 10 s shape: B=2, S≈287).

The yaml `dtype: "float32"` was kept unchanged — the production engine init
crashes when set to bfloat16, so the BF16 measurement is from the in-process
script `waterfall_bf16.py`, same setup as `waterfall_avg.py` but with
`generator.to(dtype=torch.bfloat16)` before timing.

Decoder dtype is verified at runtime: `decoder.fc2.dtype == torch.float32`.
Generator dtype: `generator.audio_heads.weight.dtype == torch.bfloat16`.

## Side-by-side per-stage

| stage | FP32 (production) | BF16 gen + FP32 dec | Δ ms | Δ % | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| embed | 1.41 | 4.32 | +2.91 | +207 % | 0.33× |
| attention | 415.13 | 399.26 | −15.87 | −3.8 % | 1.04× |
| MLP | 280.21 | 73.70 | **−206.51** | −73.7 % | **3.80×** |
| norms | 44.87 | 126.91 | **+82.04** | +183 % | 0.35× |
| audio_head | 8.88 | 1.48 | −7.40 | −83.3 % | **6.00×** |
| sampling | 23.57 | 77.51 | +53.94 | +229 % | 0.30× |
| decoder (FP32) | 9.72 | 12.52 | +2.80 | +29 % | 0.78× |
| **TOTAL** | **783.81** | **695.76** | **−88.05** | **−11.2 %** | **1.13×** |

(decoder Δ is statistical noise: per-call std on the decoder bucket is 12.78 ms.
The other deltas are real.)

## BF16 waterfall

```
  0 ─────────────────────────────────────────────────────────── 696 ms
  │
  ├── 32 × embed                       4.3 ms          █
  ├── 32 × 28 transformer layers                       ███████████████████████████████████████████████████  599.9 ms
  │       ├── attention work                                                       399.3 ms
  │       ├── MLP work                                                              73.7 ms
  │       └── norms (in + post)                                                    126.9 ms
  ├── 32 × audio_head                  1.5 ms          █
  ├── 32 × sampling                   77.5 ms          ██████
  └──  1 × decoder (FP32)             12.5 ms          █
```

## Where the speedup came from vs where it leaked

```
gross savings (matmul-heavy stages)
  MLP                       −206.5 ms
  attention                  −15.9 ms
  audio_head                  −7.4 ms
  ─────────────────────────────────
                            −229.8 ms

gross cost (mixed-precision overhead)
  norms                      +82.0 ms
  sampling                   +53.0 ms
  embed                       +2.9 ms
  decoder (noise)             +2.8 ms
  ─────────────────────────────────
                            +140.7 ms

net                          −88.1 ms   (11.2 % faster)
```

## Why the +2.9 / +82 / +53 ms costs exist

- **norms (+82 ms)**. `OmniVoiceRMSNorm.forward` has a hardcoded
  `x.to(torch.float32)` for the variance reduction. In FP32 mode it's
  optimized away (same dtype). In BF16 mode it becomes two real cast kernels
  (bf16→fp32 for variance, fp32→bf16 on the way out). Measured 37 µs extra
  per RMSNorm × 896 calls = ~33 ms inside the wrapped norms bucket, plus
  another ~50 ms in q_norm/k_norm which sit inside the wrapped `attention`
  bucket but spill across boundaries.

- **sampling (+53 ms)**. After `_get_logits` returns BF16 logits,
  `generator.forward` does `.to(torch.float32)` on a
  `[B·2, 8, S, 1025]` tensor each step — a real cast that's a no-op in FP32.
  Plus python-loop and small-op overhead becomes a larger share when GPU
  kernels shrink.

- **embed (+2.9 ms)**. Counterintuitively, the GPU does *less* work in BF16
  (21 µs vs 31 µs of kernel self-time — half the bytes through HBM for the
  embedding gathers). But the wall time grows because the 7 small kernels
  drop below PyTorch's per-op host dispatch cost (~10-15 µs), so the GPU
  finishes each kernel and waits for the next to be enqueued.

## Audio_head breakdown (kernel-level, holds across both runs)

```
FP32 kernel    : sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize64x64x8_stage3_…
BF16 kernel    : nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN
FP32 measured  : 256 µs/call ─ 36.8 TFLOPS ─ 55 % of FP32 peak (67  TFLOPS)
BF16 measured  :  17 µs/call ─ 559   TFLOPS ─ 56 % of BF16 peak (989 TFLOPS)
```

Both kernels run at ~55 % utilization of their respective peaks. The 15×
speedup at the kernel level is purely the BF16/FP32 peak ratio (989/67 ≈
14.8×), and the realized end-to-end speedup (6× on audio_head) is the
kernel-level win diluted by the per-call view/permute and ~15-20 µs of
launch overhead.

## Scripts

- `waterfall_avg.py`        — original FP32 waterfall (production dtype)
- `waterfall_bf16.py`       — BF16 generator + FP32 decoder variant
- `bench_omnivoice_10s.py`  — 10 s latency baseline
- `profile_omnivoice_buckets.py` — 5 s / 17 s / 27 s buckets
- `verify_omnivoice_timing.py`   — noise floor + linearity sanity

## Known follow-ups (not done)

- Fuse `OmniVoiceRMSNorm` (or route through `vllm_c::rms_norm`) so BF16
  doesn't pay the cast tax.
- Capture the trunk under `torch.compile(mode="reduce-overhead")` to
  collapse per-kernel launch latency that this measurement exposed.
- Drop the `.to(torch.float32)` cast after `_get_logits` — keep CFG combine
  in BF16 (log-softmax handles its own internal accumulator dtype).
- The production yaml-driven BF16 path crashes during engine init; a clean
  fix is to mirror the env-var pattern from
  `pavanyellow/vllm-omni-a100-blockwise-streaming`
  (`VLLM_OMNI_OMNIVOICE_GENERATOR_DTYPE=bf16`) so the cast happens after
  `load_weights` and only on the generator.
