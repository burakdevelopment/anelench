set -e
cd "$(dirname "$0")"
venv_python() {
  if [ -x .venv/bin/python ]; then echo .venv/bin/python; elif [ -x .venv/Scripts/python.exe ]; then echo .venv/Scripts/python.exe; fi
}
PY="$(venv_python)"
if [ -z "$PY" ]; then
  for c in python3 python py; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c "import sys; sys.exit(sys.version_info < (3, 10))" 2>/dev/null; then
      PY="$c"
      break
    fi
  done
fi
if [ -z "$PY" ]; then
  echo "Python 3.10 or newer is needed. Install it from https://www.python.org/downloads/ and run this script again."
  exit 1
fi
if ! "$PY" -c "import numpy, torch, astropy, fastapi, uvicorn" 2>/dev/null; then
  echo "Installing Anelench into a local .venv folder (first run only, a few minutes)..."
  [ -n "$(venv_python)" ] || "$PY" -m venv .venv
  PY="$(venv_python)"
  "$PY" -m pip install --upgrade pip
  if [ "$(uname -s)" = "Linux" ]; then
    "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  fi
  "$PY" -m pip install -r requirements.txt
fi
"$PY" scripts/download_eht_data.py
exec "$PY" -m anelench.server --open
