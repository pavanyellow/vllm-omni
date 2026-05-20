# OmniVoice perf investigation — handoff

Branch: `try/omnivoice-audio-head-bf16` (committed: `15165e7a`, `e2ba10c1`).

This is a learning/blog-post-driven perf investigation of the OmniVoice TTS
model in `vllm-omni`, focused on understanding where ~780 ms of wall time
for a 10 s audio generation goes and what each stage's roofline says about
why.

## What's in the branch

```
benchmarks/tts/
  AUDIO_HEAD_FP32_VS_BF16.md       findings doc: how we identified FP32 dtype
                                    from rooflines + cuBLAS kernel symbol
  BF16_GENERATOR_WATERFALL.md       findings doc: full per-stage waterfall
                                    when generator runs in BF16 + decoder FP32
  bench_omnivoice_10s.py            10 s latency baseline (5 prompts × 3 reps)
  profile_omnivoice_buckets.py      5 s / 17 s / 27 s duration buckets
  verify_omnivoice_timing.py        noise floor + linearity checks
  waterfall_avg.py                  per-stage cumulative breakdown (FP32, production)
  waterfall_bf16.py                 per-stage breakdown with BF16 generator + FP32 decoder
```

Source files in `vllm_omni/` are unmodified from `main`. The branch is
purely additive: docs and benchmark scripts only.

## Setup (next agent — run these once)

Both repos are cloned in `/root/` for cross-reference:

```
/root/OmniVoice                              # upstream k2-fsa/OmniVoice (uses HF Qwen3 directly)
/root/vllm-omni-a100-blockwise-streaming     # @pavanyellow's tuned fork
                                              # (env-var bf16, torch.compile, FA-varlen,
                                              #  bucket pre-warm, block-wise streaming)
```

Model is downloaded at `/workspace/.hf/omnivoice` (~3 GB).
Venv is at `/workspace/vllm-omni/.venv` (Python 3.12, vLLM 0.21.0, torch 2.11.0+cu130).

## How to run the baselines

```bash
# E2E latency baseline (production FP32 path, no instrumentation)
.venv/bin/python benchmarks/tts/bench_omnivoice_10s.py \
    --json /tmp/bench_baseline.json --tag baseline
# Expected: median ~768 ms ± 12 ms over 15 samples, on H100

# Duration-bucket sweep (5s / 17s / 27s)
.venv/bin/python benchmarks/tts/profile_omnivoice_buckets.py --skip-save
# Expected: ~0.6 / 1.3 / 1.9 s with RTF improving from 0.094 → 0.067

# Per-stage waterfall (production FP32)
.venv/bin/python benchmarks/tts/waterfall_avg.py
# Expected: ~784 ms total (per-component instrumented; ~118 ms of that is
#   measurement overhead from the 3.5k per-layer CUDA events)

# Per-stage waterfall with generator cast to BF16 (decoder stays FP32)
.venv/bin/python benchmarks/tts/waterfall_bf16.py
# Expected: ~696 ms total (same instrumentation overhead applies)
```

The waterfall scripts run 20 samples (5 prompts × 4 reps). Each takes
~30-60 s wall time.

## What we measured and found

### Baseline (production, FP32, instrumented per-stage)

```
0 ────────────────────────────────────────────────────────────── 784 ms
│
├── 32 × embed                  1.4 ms
├── 32 × 28 transformer layers                  740.2 ms
│       ├── attention work                      415.1 ms
│       ├── MLP work                            280.2 ms
│       └── norms (in + post)                    44.9 ms
├── 32 × audio_head             8.9 ms
├── 32 × sampling              23.6 ms
└──  1 × decoder (DAC stack)    9.7 ms (warm; 165 ms cold first call)
```

### Hardware reference (H100 80 GB HBM3)

```
FP32 peak                 :   67 TFLOPS
BF16 peak                 :  989 TFLOPS  (Hopper WGMMA tensor cores)
HBM bandwidth             :  3.0 TB/s
Kernel-launch overhead    :  ~7 µs / launch in eager mode
```

### Embed stage — what the rooflines actually say

```
Naïve FLOP estimate    : 90 GFLOPs → ~100 µs (wrong: embed is gather, not matmul)
Naïve BW estimate      : 300 MB load → ~100 µs (wrong: only ~30 rows of the
                                                 151K-entry text table are touched)
Actual FLOPs           : ~4 M (just the sum across 8 codebooks)
Actual HBM traffic     : ~23 MB (text + audio gathers + write)
Actual time            : 75.6 µs / call

Limit       Bound      Time     vs measured
compute     0.06 µs    →       1262× under
HBM BW      7.65 µs    →         9.6× under
launch      —         ~70 µs    matches: 7 kernels × ~7 µs + ~21 µs of work
```

The embed stage is **kernel-launch bound**, not compute or BW bound. The
gather table being huge doesn't matter because `nn.Embedding` only reads
the rows it's asked for.

### Audio_head — the "model is in FP32" smoking gun

`audio_head` is a single `Linear(1024 → 8·1025)` matmul, called once per
unmasking step (×32 per request).

```
BF16 roofline   : 9.5 µs / call,   0.30 ms × 32 steps
FP32 roofline   : 140 µs / call,   4.49 ms × 32 steps
Measured        : 273 µs / call,   8.74 ms × 32 steps

→ 27× over BF16 roofline   → impossible efficiency, must not be BF16
→  1.94× over FP32 roofline → consistent with 55 % of FP32 peak
```

cuBLAS picked `sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize64x64x8_…`
(an Ampere-era kernel — H100 falls back to it for FP32). The
`f32f32_f32f32_f32` slug literally encodes (A, B, accum, C, output) dtypes.
The deploy yaml hard-codes `dtype: "float32"`.

Running the same matmul in BF16 picks
`nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN` (Hopper-native WGMMA),
17 µs/call — **15× faster**, both at ~55 % of their respective peak.

### BF16 generator (decoder kept FP32) — full waterfall

```
0 ─────────────────────────────────────────────────────────── 696 ms
│
├── 32 × embed                  4.3 ms
├── 32 × 28 transformer layers                  599.9 ms
│       ├── attention work                      399.3 ms
│       ├── MLP work                             73.7 ms
│       └── norms (in + post)                   126.9 ms
├── 32 × audio_head             1.5 ms
├── 32 × sampling              77.5 ms
└──  1 × decoder (FP32)        12.5 ms
```

**11.2 % e2e speedup** (784 → 696 ms). The matmul wins (MLP −207 ms,
audio_head −7 ms) are partly eaten by:

- `norms` regress +82 ms — `OmniVoiceRMSNorm` always casts to FP32 for
  variance (`x.to(torch.float32).pow(2).mean(...)`), which becomes two
  real cast kernels in BF16 mode.
- `sampling` regresses +54 ms — the `.to(torch.float32)` cast on logits
  after `_get_logits` is a real cast (no-op in FP32), plus short BF16
  kernels expose more launch latency.
- `embed` regresses +3 ms — counterintuitive: GPU work drops 33 %
  (21 µs vs 31 µs of kernel time) because BF16 halves the bytes
  through HBM. Wall time goes *up* because the kernels become so
  short that they drop below PyTorch's per-op host dispatch cost
  (~10-15 µs), so the GPU finishes each kernel and waits.

### One important methodology caveat

The 696 ms and 784 ms numbers in the waterfalls above are **instrumented**
wall times — the per-layer CUDA-event wraps add ~118 ms of measurement
overhead. The **uninstrumented** wall (production engine) is closer to:

```
FP32 minimal wraps:  ~666 ms  (extrapolated from the 118 ms gap)
BF16 minimal wraps:   577 ms  (measured directly)
```

Both the FP32 and BF16 waterfalls use the same wrap depth, so the
relative comparison (−11 % speedup, breakdown by stage) is fair. The
absolute numbers are inflated.

### Decoder cold start surprise

```
Cold first call (after weights load) :  165 ms     ← cuDNN autotuning
Warm steady state (×50 reps)         :  5.90 ms    ← 28× faster
Std across warm calls                :  0.05 ms
```

The decoder is dominated by the DAC `ConvTranspose1d` stack doing
960× temporal upsampling (245 frames → 235 200 samples). cuDNN picks an
algorithm per shape on the first call and caches it; from call 2 onward,
it's effectively a constant ~6 ms.

## Commands that produced each artifact

```bash
# Audio_head kernel symbol (for both precisions)
.venv/bin/python -c "...torch.profiler around gen._get_logits..."
# (script lives in commit history of e2ba10c1 — see AUDIO_HEAD_FP32_VS_BF16.md)

# Decoder direct timing (cold/warm split, sub-stage breakdown)
.venv/bin/python /tmp/decoder_direct.py   # see commit message for layout

# Embed-stage micro-bench (10 trials × 500 iters, FP32 vs BF16)
.venv/bin/python /tmp/measure_embed_op.py
```

The `/tmp/*.py` throwaway scripts are not in the repo — they're written
by the previous agent in scratch space and lost on restart. The
production-quality versions live in `benchmarks/tts/`.

## Reference repos (cloned locally)

- `/root/OmniVoice` — upstream `k2-fsa/OmniVoice` (uses HF `Qwen3Model`
  via `AutoModel.from_config`, gets FA2/SDPA/FlexAttn for free).
- `/root/vllm-omni-a100-blockwise-streaming` — production-tuned fork.
  Has `VLLM_OMNI_OMNIVOICE_GENERATOR_DTYPE=bf16` env var (clean cast
  pattern), `torch.compile` per layer, FA-varlen with precomputed
  metadata, bucket pre-warm for cudagraphs, block-wise streaming.
- `/root/vllm-omni` (this repo, on `try/omnivoice-audio-head-bf16`) —
  unmodified hand-roll of Qwen3.
