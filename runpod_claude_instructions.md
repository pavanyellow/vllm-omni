# vllm-omni on RunPod H100 — environment setup

System basics (gh CLI, uv, this repo at `/workspace/vllm-omni`) are already
installed by `runpod_setup.sh`. The steps below get the Python env, deps, and
OmniVoice checkpoint in place.

## 1. Python venv

```bash
cd /workspace/vllm-omni
uv venv .venv --python 3.11
source .venv/bin/activate
```

## 2. Install vllm-omni in editable mode

Read `pyproject.toml` first to see what's declared. Then:

```bash
uv pip install -e .
```

If the vllm dep fails to resolve, install it explicitly for CUDA 12.4:
```bash
uv pip install vllm --extra-index-url https://download.pytorch.org/whl/cu124
uv pip install -e .
```

## 3. OmniVoice extra deps

```bash
uv pip install \
    "transformers>=5.3" \
    "tokenizers" \
    "torchaudio" \
    "soundfile" \
    "librosa" \
    "huggingface_hub"
```

`transformers>=5.3` is required for `HiggsAudioV2TokenizerModel` (voice
cloning) — see
`vllm_omni/diffusion/models/omnivoice/pipeline_omnivoice.py:36-38`.

## 4. Fetch the OmniVoice checkpoint

```bash
huggingface-cli download k2-fsa/OmniVoice
```

~2 GB. Lands in `$HF_HOME/hub` which is on the persistent volume.
