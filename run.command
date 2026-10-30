#!/bin/bash
# Double-click from Finder (macOS), or run ./run.command [options] from a terminal.
# Creates the Python environment on first launch, then starts GokuCam.
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  PY=$(command -v python3.14 || command -v python3.13 || command -v python3.12 || command -v python3.11 || command -v python3.10)
  if [ -z "$PY" ]; then
    echo "Python 3.10+ is required (e.g. brew install python@3.13)"; read -r; exit 1
  fi
  "$PY" -m venv .venv && .venv/bin/pip install -r requirements.txt || { read -r; exit 1; }
fi
exec .venv/bin/python -m gokucam "$@"
