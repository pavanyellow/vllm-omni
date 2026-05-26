# Unified voice-cloning TTS API (Qwen3-TTS + OmniVoice)

One HTTP API, cloning-first, fronting **two** vLLM-Omni TTS models. Upload a
reference clip once to get a `voice_id`, then synthesize any text in that voice
with **either** model:

- **qwen3** — `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (voice-cloning checkpoint)
- **omnivoice** — `k2-fsa/OmniVoice` (zero-shot multilingual cloning)

OmniVoice cloning is driven through the **offline `Omni` engine**, because the
stock vllm-omni HTTP server only exposes OmniVoice "auto voice".

## One script does everything

`tts_api.sh` is fully self-contained — it installs deps, builds the venvs,
downloads the models, writes the worker/gateway code, and launches the stack.
On a fresh GPU pod:

```bash
bash tts_api.sh            # setup (if needed) + start
```

Subcommands:

```bash
bash tts_api.sh setup      # install deps + venvs + models only
bash tts_api.sh start      # start workers + gateway (refuses if already up)
bash tts_api.sh restart    # stop then start
bash tts_api.sh stop       # stop everything
bash tts_api.sh status     # gateway + worker health
bash tts_api.sh test       # clone a voice + generate on both models
bash tts_api.sh docs       # print the full API reference
```

`worker.py` and `gateway.py` are committed here for readability; `tts_api.sh`
also embeds them verbatim and writes them to `/root/tts/` at runtime, so the
single script is enough on its own.

## Architecture

```
laptop --ssh tunnel--> :8000 gateway ──> :8101 qwen3 worker      (Omni engine)
                       (voice registry) └─> :8102 omnivoice worker (Omni engine)
```

Each model runs in its own process (and its own venv — see below), so they
can't destabilize each other; GPU memory is split (qwen3 ~0.6, omnivoice ~0.3
of the GPU).

## Reach it from your laptop (SSH port-forward)

The gateway binds `0.0.0.0:8000`; the per-model workers stay on localhost. Use
your pod's direct-TCP SSH endpoint (the proxy `ssh.runpod.io` form does **not**
support `-L`):

```bash
ssh -N -L 8000:localhost:8000 root@<pod-ip> -p <ssh-tcp-port> -i ~/.ssh/id_ed25519
# leave it running; then in another terminal:
curl http://localhost:8000/health
```

If local port 8000 is taken, forward to another local port, e.g.
`-L 8080:localhost:8000`, and hit `http://localhost:8080`.

## API

### 1. Clone a voice → `voice_id`

```bash
curl -X POST http://localhost:8000/voices \
  -F "audio_sample=@/path/to/reference.wav" \
  -F "ref_text=The exact transcript spoken in the reference clip." \
  -F "name=my_voice"
# -> {"voice_id":"ab12cd34ef56", ...}
```

3–30 s of clean single-speaker audio clones best.

### 2. Generate with that voice (either model)

```bash
# Qwen3-TTS
curl -X POST http://localhost:8000/tts -H "Content-Type: application/json" \
  -d '{"text":"Hello, this is my cloned voice.","voice_id":"ab12cd34ef56","model":"qwen3","language":"English"}' \
  --output out_qwen3.wav

# OmniVoice (same voice_id)
curl -X POST http://localhost:8000/tts -H "Content-Type: application/json" \
  -d '{"text":"Bonjour, ceci est ma voix clonée.","voice_id":"ab12cd34ef56","model":"omnivoice","language":"fr"}' \
  --output out_omni.wav
```

Response is a 16-bit mono 24 kHz WAV. Headers: `X-Model`, `X-Voice-Id`,
`X-Sample-Rate`, `X-Duration-Sec`, `X-Gen-Sec`.

### Endpoints

| method | path | purpose |
|---|---|---|
| GET | `/health` | gateway + worker readiness |
| POST | `/voices` | multipart upload → register/clone a voice |
| GET | `/voices` | list registered voices |
| GET | `/voices/{id}` | voice detail |
| DELETE | `/voices/{id}` | remove a voice |
| POST | `/tts` | synthesize (JSON) → `audio/wav` |

### `POST /tts` fields

| field | required | notes |
|---|---|---|
| `text` | yes | text to synthesize |
| `voice_id` | yes | from `POST /voices` |
| `model` | no | `qwen3` (default) or `omnivoice` |
| `language` | no | qwen3: `English`/`Chinese`/`Auto`/… · omnivoice: `en`/`zh`/… |
| `instruct` | no | omnivoice voice-design hint, e.g. `"female, low pitch"` |
| `max_new_tokens` | no | default 2048 |

## Notes & gotchas

- **No streaming** — each request returns a complete WAV.
- A `voice_id` is just the stored reference clip + transcript, so it works with
  either model.
- **Latency** (A100, warm): omnivoice ~2 s; qwen3 ~16 s. The first qwen3 request
  after startup is ~100 s (one-time JIT compile).
- Requests are serialized per model (one generation at a time) — fine for
  personal use, not tuned for concurrency.

### Why two venvs

vLLM 0.21 forbids transformers 5.0–5.4. Qwen3-TTS's vendored decoder needs the
pre-rename `create_causal_mask(input_embeds=...)` (transformers 4.57), while
OmniVoice cloning needs `HiggsAudioV2TokenizerModel` (transformers ≥5.3). No
single version satisfies both, so each model gets its own venv:

- `/root/tts-venv` — transformers 5.9 — OmniVoice worker + gateway
- `/root/tts-venv-qwen3` — transformers 4.57.6 — Qwen3 worker

`tts_api.sh` also puts each venv's `bin` on `PATH` (vLLM's JIT compile needs the
venv's `ninja` on `PATH`, not merely installed). On RunPod it forces uv/HF
caches onto the fast local overlay disk rather than the slow `/workspace`
network mount.
