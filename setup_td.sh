#!/usr/bin/env bash
# setup_td.sh — Create a TouchDesigner-compatible Python environment.
#
# Why this is separate from setup.sh:
#   TouchDesigner ships with its own Python 3.11 interpreter.  Native
#   extensions (cv2, mediapipe, torch, numpy, scipy) MUST match TD's ABI,
#   so we need a parallel venv built specifically against Python 3.11 — the
#   default `.venv` from setup.sh may be a newer version (e.g. 3.14).
#
# What this script does:
#   1. Verifies Python 3.11 is installed; offers to `brew install python@3.11`.
#   2. Creates `.venv_td` in the project root using python3.11.
#   3. Installs the same requirements.txt into it.
#   4. Prints the absolute site-packages path you must paste into:
#        TouchDesigner → Edit → Preferences → DATs → Python 64-bit Module Path

set -euo pipefail

GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[1;33m"
NC="\033[0m"

info()  { echo -e "${GREEN}[setup_td]${NC} $*"; }
warn()  { echo -e "${YELLOW}[setup_td]${NC} $*"; }
error() { echo -e "${RED}[setup_td]${NC} $*" >&2; }
die()   { error "$*"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv_td"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"

# ---------------------------------------------------------------------------
# 1. Locate Python 3.11
# ---------------------------------------------------------------------------
PY311=""
for candidate in \
        "/opt/homebrew/opt/python@3.11/bin/python3.11" \
        "/usr/local/opt/python@3.11/bin/python3.11" \
        "$(command -v python3.11 || true)"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
        PY311="$candidate"
        break
    fi
done

if [ -z "$PY311" ]; then
    warn "Python 3.11 not found."
    if command -v brew &>/dev/null; then
        read -r -p "Install via 'brew install python@3.11'? [y/N] " ans
        if [[ "$ans" =~ ^[Yy]$ ]]; then
            brew install python@3.11
            PY311="$(/opt/homebrew/opt/python@3.11/bin/python3.11 -c 'import sys;print(sys.executable)' 2>/dev/null || true)"
        fi
    fi
    [ -z "$PY311" ] && die "Install Python 3.11 manually, then re-run."
fi
info "Using $PY311 ($($PY311 --version))"

# ---------------------------------------------------------------------------
# 2. Create the venv
# ---------------------------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
    info "Creating venv at $VENV_DIR..."
    "$PY311" -m venv "$VENV_DIR"
else
    info "Reusing existing venv at $VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# ---------------------------------------------------------------------------
# 3. Install requirements
# ---------------------------------------------------------------------------
info "Upgrading pip..."
pip install --quiet --upgrade pip

[ -f "$REQUIREMENTS" ] || die "requirements.txt missing at $REQUIREMENTS"
info "Installing dependencies into the TD venv (this may take a few minutes)..."
pip install --quiet -r "$REQUIREMENTS"

# ---------------------------------------------------------------------------
# 4. Print the site-packages path for TD's Python Module Path preference
# ---------------------------------------------------------------------------
SITE_PKGS="$VENV_DIR/lib/python3.11/site-packages"

echo ""
info "TD-compatible venv ready."
echo ""
echo -e "${YELLOW}Paste this path into TouchDesigner:${NC}"
echo -e "  ${GREEN}Edit → Preferences → DATs → Python 64-bit Module Path${NC}"
echo ""
echo "  $SITE_PKGS"
echo ""
echo "Then restart TouchDesigner."
