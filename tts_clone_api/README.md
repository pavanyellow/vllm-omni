# Unified voice-cloning TTS API (Qwen3-TTS + OmniVoice)

One cloning-first HTTP API over two models: **qwen3** (`Qwen/Qwen3-TTS-12Hz-1.7B-Base`)
and **omnivoice** (`k2-fsa/OmniVoice`). Upload a reference clip → get a `voice_id`
→ synthesize any text in that voice with either model.

## Run

`tts_api.sh` is self-contained (installs deps, builds venvs, downloads models,
launches). On a fresh GPU pod:

```bash
bash tts_api.sh            # setup (if needed) + start, serves on :8000
```

```bash
bash tts_api.sh stop | restart | status | test | docs
```

## Reach it from your laptop

```bash
ssh -N -L 8000:localhost:8000 root@<pod-ip> -p <ssh-tcp-port> -i ~/.ssh/id_ed25519
# leave running; then:
curl http://localhost:8000/health
```

## API

### Clone a voice → `voice_id`

```bash
curl -X POST http://localhost:8000/voices \
  -F "audio_sample=@/path/to/reference.wav" \
  -F "ref_text=The exact transcript spoken in the clip." \
  -F "name=my_voice"
# -> {"voice_id":"ab12cd34ef56", ...}
```

### Generate (either model, same `voice_id`)

```bash
curl -X POST http://localhost:8000/tts -H "Content-Type: application/json" \
  -d '{"text":"Hello, this is my cloned voice.","voice_id":"ab12cd34ef56","model":"qwen3","language":"English"}' \
  --output out.wav
```

Returns a 16-bit mono 24 kHz WAV.

### Endpoints

| method | path | purpose |
|---|---|---|
| GET | `/health` | gateway + worker readiness |
| POST | `/voices` | upload clip → register/clone a voice |
| GET | `/voices` | list voices |
| GET | `/voices/{id}` | voice detail |
| DELETE | `/voices/{id}` | remove a voice |
| POST | `/tts` | synthesize → `audio/wav` |

### `POST /tts` fields

| field | required | notes |
|---|---|---|
| `text` | yes | text to synthesize |
| `voice_id` | yes | from `POST /voices` |
| `model` | no | `qwen3` (default) or `omnivoice` |
| `language` | no | qwen3: `English`/`Chinese`/`Auto`/… · omnivoice: `en`/`zh`/… |
| `instruct` | no | omnivoice voice-design hint, e.g. `"female, low pitch"` |
| `max_new_tokens` | no | default 2048 |
