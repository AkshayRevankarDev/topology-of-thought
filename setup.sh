#!/usr/bin/env bash
# setup.sh — Bootstrap Topology of Thought development environment.
#
# What this script does:
#   1. Verifies Python 3.11+ is available.
#   2. Creates a virtual environment (optional, activated if already present).
#   3. Upgrades pip and installs all Python dependencies.
#   4. Pulls the llama3.2 model via Ollama.
#   5. Creates required data subdirectories.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# To use inside an existing venv, activate it first, then run this script.

set -euo pipefail

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[1;33m"
NC="\033[0m"  # No colour

info()    { echo -e "${GREEN}[setup]${NC} $*"; }
warn()    { echo -e "${YELLOW}[setup]${NC} $*"; }
error()   { echo -e "${RED}[setup]${NC} $*" >&2; }
die()     { error "$*"; exit 1; }

# ---------------------------------------------------------------------------
# 1. Python version check
# ---------------------------------------------------------------------------
info "Checking Python version..."
PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" &>/dev/null; then
    die "Python not found. Install Python 3.11+ and ensure it is on PATH."
fi

PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
    die "Python 3.11+ is required. Found: $PY_VERSION"
fi
info "Python $PY_VERSION OK."

# ---------------------------------------------------------------------------
# 2. Virtual environment (optional — skip if already inside one)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ ! -d "$VENV_DIR" ]; then
        info "Creating virtual environment at $VENV_DIR..."
        "$PYTHON" -m venv "$VENV_DIR"
    fi
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    info "Virtual environment activated."
else
    info "Already inside a virtual environment: $VIRTUAL_ENV"
fi

# ---------------------------------------------------------------------------
# 3. Upgrade pip and install dependencies
# ---------------------------------------------------------------------------
info "Upgrading pip..."
pip install --quiet --upgrade pip

REQUIREMENTS="$SCRIPT_DIR/requirements.txt"
if [ ! -f "$REQUIREMENTS" ]; then
    die "requirements.txt not found at $REQUIREMENTS"
fi

info "Installing Python dependencies (this may take a few minutes)..."
pip install --quiet -r "$REQUIREMENTS"
info "Python dependencies installed."

# ---------------------------------------------------------------------------
# 4. Pull llama3.2 via Ollama
# ---------------------------------------------------------------------------
info "Checking for Ollama..."
if ! command -v ollama &>/dev/null; then
    warn "Ollama CLI not found on PATH."
    warn "Install Ollama from https://ollama.com/download, then run:"
    warn "    ollama pull llama3.2"
else
    info "Pulling llama3.2 model via Ollama (requires Ollama server running)..."
    if ollama pull llama3.2; then
        info "llama3.2 model ready."
    else
        warn "ollama pull llama3.2 failed. Start Ollama and run it manually:"
        warn "    ollama pull llama3.2"
    fi
fi

# ---------------------------------------------------------------------------
# 5. Create data subdirectories
# ---------------------------------------------------------------------------
info "Creating data directories..."
DATA_DIR="$SCRIPT_DIR/data"
mkdir -p "$DATA_DIR/sessions"
mkdir -p "$DATA_DIR/exports"
mkdir -p "$DATA_DIR/pdfs"
info "Data directories ready:"
info "  $DATA_DIR/sessions  — saved JSON graph states"
info "  $DATA_DIR/exports   — Anki .apkg files"
info "  $DATA_DIR/pdfs      — drop PDFs here"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
info "Setup complete!"
info "Next steps:"
info "  1. Ensure Ollama is running:  ollama serve"
info "  2. Run health checks:         python health_check.py"
info "  3. Run pipeline test:         python pipeline_test.py"
