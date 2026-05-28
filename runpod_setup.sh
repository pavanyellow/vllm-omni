#!/usr/bin/env bash
# RunPod setup for vllm-omni OmniVoice optimization (H100 SXM).
#
# Usage on a fresh RunPod pod:
#   wget https://raw.githubusercontent.com/pavanyellow/vllm-omni/main/runpod_setup.sh
#   bash runpod_setup.sh
#
# After it finishes:
#   cd /workspace/vllm-omni && claude
#   then ask Claude to follow runpod_claude_instructions.md
#
# Recommended pod image:
#   runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

set -euo pipefail

WORKSPACE=${WORKSPACE:-/workspace}
REPO_URL=${REPO_URL:-https://github.com/vllm-project/vllm-omni.git}
REPO_DIR="$WORKSPACE/vllm-omni"

echo "==> vllm-omni RunPod setup"
echo "==> Workspace: $WORKSPACE"
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

# ── GPU sanity ───────────────────────────────────────────────────────────────
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: no nvidia-smi — pick a CUDA RunPod template." >&2
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# ── System basics ────────────────────────────────────────────────────────────
echo "==> apt: installing basics"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
    curl wget git build-essential ca-certificates gnupg \
    htop tmux vim less jq tree \
    libsndfile1 ffmpeg \
    > /dev/null

# ── GitHub CLI ───────────────────────────────────────────────────────────────
if ! command -v gh &>/dev/null; then
    echo "==> Installing GitHub CLI"
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg 2>/dev/null
    chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list
    apt-get update -qq
    apt-get install -y -qq gh >/dev/null
fi
gh --version | head -1

# ── Claude Code ──────────────────────────────────────────────────────────────
if ! command -v claude &>/dev/null; then
    echo "==> Installing Claude Code"
    curl -fsSL https://claude.ai/install.sh | bash
fi
export PATH="$HOME/.local/bin:$PATH"
claude --version || echo "(Claude installed — open a new shell or run: export PATH=\$HOME/.local/bin:\$PATH)"

# ── Node.js (needed for Codex CLI) ───────────────────────────────────────────
if ! command -v node &>/dev/null || [ "$(node -v 2>/dev/null | sed 's/v//;s/\..*//')" -lt 22 ]; then
    echo "==> Installing Node.js 22"
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
    apt-get install -y -qq nodejs >/dev/null
fi
node -v

# ── Codex CLI ────────────────────────────────────────────────────────────────
if ! command -v codex &>/dev/null; then
    echo "==> Installing Codex CLI"
    npm install -g @openai/codex
fi
codex --version || true

# ── uv (fast Python package manager) ─────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "==> Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

# ── Clone repo ───────────────────────────────────────────────────────────────
if [ ! -d "$REPO_DIR" ]; then
    echo "==> Cloning vllm-omni → $REPO_DIR"
    git clone "$REPO_URL" "$REPO_DIR"
else
    echo "==> Repo already at $REPO_DIR (skipping clone)"
fi

# ── Persistent env vars in bashrc ────────────────────────────────────────────
BASHRC="$HOME/.bashrc"
add_env() {
    grep -qF "$1" "$BASHRC" 2>/dev/null || echo "$1" >> "$BASHRC"
}
add_env 'export PATH="$HOME/.local/bin:$PATH"'
add_env "export HF_HOME=$WORKSPACE/.hf"
add_env "export HUGGINGFACE_HUB_CACHE=$WORKSPACE/.hf/hub"
add_env "export VLLM_OMNI_HOME=$REPO_DIR"
add_env "export IS_SANDBOX=1"
add_env "alias claude='claude --dangerously-skip-permissions'"
add_env "alias codex='codex --yolo'"
mkdir -p "$WORKSPACE/.hf/hub"
export HF_HOME="$WORKSPACE/.hf"
export HUGGINGFACE_HUB_CACHE="$WORKSPACE/.hf/hub"

# ── Done ─────────────────────────────────────────────────────────────────────
cat <<EOF

================================================================================
System setup complete.

Next steps:
  1. (optional) gh auth login                     # for PRs
  2. (optional) export HF_TOKEN=hf_xxx            # if any model is gated
  3. exec bash                                    # reload bashrc, picks up PATH
  4. cd $REPO_DIR && claude
     → ask Claude to follow runpod_claude_instructions.md

================================================================================
EOF
