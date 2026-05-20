# Averaged waterfall for OmniVoice 10s requests.
# Runs N samples, each over a different ~10s prompt, instruments each request
# at the (embed / attention / MLP / norms / audio_head / sampling / decoder)
# level, and reports mean ± std per stage.
#
# Instrumentation: each layer's input_LN, self_attn, post_LN, and mlp are
# wrapped with CUDA-event timers. ~3.5k events per request, ~2% overhead.

import json, os, statistics, torch
from collections import defaultdict
from tokenizers import Tokenizer as HFTokenizer

from vllm_omni.model_executor.models.omnivoice.omnivoice_generator import OmniVoiceGenerator
from vllm_omni.model_executor.models.omnivoice.omnivoice_decoder import OmniVoiceDecoder
from vllm_omni.model_executor.models.omnivoice.duration import RuleDurationEstimator
from vllm_omni.transformers_utils.configs.omnivoice import OmniVoiceConfig

MODEL_DIR = "/workspace/.hf/omnivoice"
device = torch.device("cuda")

# Five 10s prompts from the bench
PROMPTS = [
    ("Good morning everyone, thanks for joining the call today. We will start with a quick "
     "recap of last week and then walk through the latest results before we wrap up."),
    ("The library was completely silent except for the soft hum of the air conditioning and "
     "the occasional turn of a page as students prepared for the long final exam week."),
    ("When you open the application for the first time please grant microphone access, "
     "sign in with your company email, and complete the short tutorial that follows it."),
    ("Speech synthesis quality depends not only on the acoustic model but also on the text "
     "frontend, the duration model, and the neural vocoder used at the end of the pipeline."),
    ("Travel updates for the morning commute: trains on the red line are running about ten "
     "minutes late because of a signal problem near the downtown interchange this morning."),
]
REPS = 4   # 5 prompts × 4 reps = 20 samples
N_SAMPLES = len(PROMPTS) * REPS

# --- build pipeline ---------------------------------------------------------
with open(os.path.join(MODEL_DIR, "config.json")) as f:
    config = OmniVoiceConfig(**json.load(f))
generator = OmniVoiceGenerator(config).eval()
decoder = OmniVoiceDecoder(config).eval()
generator.load_weights(MODEL_DIR, device)
generator = generator.to(device=device, dtype=torch.float32).eval()
decoder.load_weights(MODEL_DIR, device)
tok = HFTokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
dur = RuleDurationEstimator()

# --- shared instrumentation -------------------------------------------------
events_log = []   # list of (stage_name, start_event, end_event)

def wrap(name, fn):
    def wrapped(*a, **kw):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = fn(*a, **kw)
        e.record()
        events_log.append((name, s, e))
        return out
    return wrapped

# top-level
generator._prepare_embeddings = wrap("embed",     generator._prepare_embeddings)
generator._get_logits         = wrap("aud_head",  generator._get_logits)
decoder.forward               = wrap("decoder",   decoder.forward)

# per-layer pieces
for blk in generator.layers:
    blk.input_layernorm.forward          = wrap("norm_in",  blk.input_layernorm.forward)
    blk.self_attn.forward                 = wrap("attn",    blk.self_attn.forward)
    blk.post_attention_layernorm.forward = wrap("norm_pst", blk.post_attention_layernorm.forward)
    blk.mlp.forward                       = wrap("mlp",     blk.mlp.forward)

# --- helper: build inputs for a given prompt --------------------------------
NUM_CB, MASK_ID = config.num_audio_codebook, config.audio_mask_id

def make_inputs(text: str):
    fp = (f"<|denoise|><|lang_start|>None<|lang_end|>"
          f"<|instruct_start|>None<|instruct_end|><|text_start|>{text}<|text_end|>")
    text_ids = torch.tensor(tok.encode(fp).ids, dtype=torch.long, device=device)
    text_len = text_ids.shape[0]
    target_len = max(1, int(dur.estimate_duration(text, "Nice to meet you.", 25)))
    text_b = text_ids.unsqueeze(0).repeat(NUM_CB, 1)
    tgt = torch.full((NUM_CB, target_len), MASK_ID, dtype=torch.long, device=device)
    cond = torch.cat([text_b, tgt], dim=1)
    cl = cond.shape[1]
    unc = tgt.clone()
    max_len = cl
    if target_len < max_len:
        pad = torch.full((NUM_CB, max_len - target_len), MASK_ID, dtype=torch.long, device=device)
        unc = torch.cat([unc, pad], dim=1)
    ids = torch.stack([cond, unc])
    am = torch.zeros(2, max_len, dtype=torch.bool, device=device)
    am[0, text_len:cl] = True
    am[1, :target_len] = True
    attm = torch.zeros(2, 1, max_len, max_len, dtype=torch.bool, device=device)
    attm[0, :, :cl, :cl] = True
    attm[1, :, :target_len, :target_len] = True
    return ids, am, attm, target_len

# --- warmup (5 calls; clears any cold-start effects on later samples) -------
print("warming up...")
warm_ids, warm_am, warm_attm, warm_tl = make_inputs(PROMPTS[0])
for _ in range(3):
    events_log.clear()
    with torch.inference_mode():
        tokens = generator(warm_ids, warm_am, warm_attm, [warm_tl],
                           num_step=config.num_step, guidance_scale=config.guidance_scale,
                           t_shift=config.t_shift, layer_penalty_factor=config.layer_penalty_factor,
                           position_temperature=config.position_temperature, class_temperature=config.class_temperature)
        _ = decoder(tokens)
torch.cuda.synchronize()

# --- timed runs --------------------------------------------------------------
samples = []
# interleave so any drift hits all prompts equally: P0,P1,P2,P3,P4,P0,P1,...
schedule = [(rep, idx) for rep in range(REPS) for idx in range(len(PROMPTS))]
for sample_idx, (rep, pidx) in enumerate(schedule):
    prompt = PROMPTS[pidx]
    ids, am, attm, tl = make_inputs(prompt)
    events_log.clear()

    GS = torch.cuda.Event(enable_timing=True); GE = torch.cuda.Event(enable_timing=True)
    DS = torch.cuda.Event(enable_timing=True); DE = torch.cuda.Event(enable_timing=True)
    GS.record()
    with torch.inference_mode():
        tokens = generator(ids, am, attm, [tl],
                           num_step=config.num_step, guidance_scale=config.guidance_scale,
                           t_shift=config.t_shift, layer_penalty_factor=config.layer_penalty_factor,
                           position_temperature=config.position_temperature, class_temperature=config.class_temperature)
    GE.record()
    DS.record()
    with torch.inference_mode():
        audio = decoder(tokens)
    DE.record()
    torch.cuda.synchronize()

    gen_ms = GS.elapsed_time(GE)
    dec_ms = DS.elapsed_time(DE)

    # aggregate events by stage name
    agg = defaultdict(float)
    cnt = defaultdict(int)
    for n, s, e in events_log:
        agg[n] += s.elapsed_time(e)   # ms
        cnt[n] += 1

    # roll up
    embed_ms     = agg["embed"]
    norm_in_ms   = agg["norm_in"]
    attn_ms      = agg["attn"]
    norm_pst_ms  = agg["norm_pst"]
    mlp_ms       = agg["mlp"]
    norms_ms     = norm_in_ms + norm_pst_ms
    aud_head_ms  = agg["aud_head"]
    decoder_ms   = agg["decoder"]
    # sampling = gen_ms - (embed + sum(layer pieces) + audio_head)
    layer_sum    = norm_in_ms + attn_ms + norm_pst_ms + mlp_ms
    sampling_ms  = gen_ms - embed_ms - layer_sum - aud_head_ms
    total_ms     = gen_ms + dec_ms
    audio_s      = audio.shape[-1] / config.sample_rate

    samples.append({
        "prompt_idx": pidx,
        "rep":        rep,
        "audio_s":   audio_s,
        "total_ms":  total_ms,
        "embed":     embed_ms,
        "attn":      attn_ms,
        "mlp":       mlp_ms,
        "norms":     norms_ms,
        "aud_head":  aud_head_ms,
        "sampling":  sampling_ms,
        "decoder":   decoder_ms,
    })
    print(f"  rep={rep} p{pidx}: audio={audio_s:5.2f}s  wall={total_ms:7.2f} ms")

# --- aggregate stats --------------------------------------------------------
def stat(rows, key):
    vals = [r[key] for r in rows]
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

print(f"\n=== averaged over {N_SAMPLES} samples (5 prompts × 1 rep) ===\n")
print(f"  mean audio output: {statistics.mean(r['audio_s'] for r in samples):.2f} s")
print(f"  mean wall total:   {statistics.mean(r['total_ms'] for r in samples):.2f} ms")
print(f"  std  wall total:   {statistics.stdev(r['total_ms'] for r in samples):.2f} ms\n")

stages = [
    ("embed",     "32 × _prepare_embeddings"),
    ("attn",      "896 × attention block (q+k+v+norms+RoPE+SDPA+o)"),
    ("mlp",       "896 × MLP block (gate+up+silu·mul+down)"),
    ("norms",     "896 × 2 RMSNorms (input + post)"),
    ("aud_head",  "32 × audio_heads projection"),
    ("sampling",  "32 × CFG/log-softmax/gumbel/topk/scatter"),
    ("decoder",   "1 × decoder (RVQ + fc2 + DAC stack)"),
]

print(f"  {'stage':<14} {'mean ms':>10} {'std ms':>8} {'% total':>8}  description")
print(f"  {'-'*14} {'-'*10} {'-'*8} {'-'*8}  {'-'*48}")
total_mean = statistics.mean(r['total_ms'] for r in samples)
for key, desc in stages:
    m, sd = stat(samples, key)
    pct = m / total_mean * 100
    print(f"  {key:<14} {m:>9.2f}  {sd:>7.2f}  {pct:>6.2f}%   {desc}")
print(f"  {'-'*14} {'-'*10} {'-'*8} {'-'*8}")
print(f"  {'TOTAL':<14} {total_mean:>9.2f}  {statistics.stdev(r['total_ms'] for r in samples):>7.2f}  100.00%")

# --- waterfall ASCII (means) ------------------------------------------------
print(f"\n  Full request, averaged waterfall  (mean over {N_SAMPLES} samples)\n")
print(f"  0 ────────────────────────────────────────────────────────────── {total_mean:.0f} ms")
print(f"  │")

def bar(ms):
    width = max(1, int(ms / total_mean * 65))
    return "█" * width

attn_m, _ = stat(samples, "attn")
mlp_m, _  = stat(samples, "mlp")
norm_m, _ = stat(samples, "norms")
trunk_m = attn_m + mlp_m + norm_m

print(f"  ├── 32 × embed                  {samples[0]['embed']:.1f} ms".ljust(50) + bar(stat(samples, 'embed')[0]))
print(f"  ├── 32 × 28 transformer layers".ljust(50) + bar(trunk_m) + f"  {trunk_m:.1f} ms")
print(f"  │       ├── attention work".ljust(50) + " " * 30 + f"{attn_m:.1f} ms")
print(f"  │       ├── MLP work".ljust(50) + " " * 30 + f"{mlp_m:.1f} ms")
print(f"  │       └── norms (in + post)".ljust(50) + " " * 30 + f"{norm_m:.1f} ms")
print(f"  ├── 32 × audio_head             {stat(samples, 'aud_head')[0]:.1f} ms".ljust(50) + bar(stat(samples, 'aud_head')[0]))
print(f"  ├── 32 × sampling               {stat(samples, 'sampling')[0]:.1f} ms".ljust(50) + bar(stat(samples, 'sampling')[0]))
print(f"  └──  1 × decoder (DAC stack)".ljust(50) + bar(stat(samples, 'decoder')[0]) + f"  {stat(samples, 'decoder')[0]:.1f} ms")
