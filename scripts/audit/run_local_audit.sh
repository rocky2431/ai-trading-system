#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

WITH_TESTS="${1:-}"

echo "== Repo =="
git rev-parse HEAD
git rev-parse --abbrev-ref HEAD
git status --porcelain=v1 -b
echo

echo "== Python =="
python -V || true
echo "python_executable=$(python -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
echo

echo "== Qlib (deep fork check) =="
PYTHONPATH=src:vendor/qlib python - <<'PY' || true
import qlib
print("qlib.__file__=", qlib.__file__)
print("qlib.__version__=", getattr(qlib, "__version__", None))
try:
    from qlib.constant import REG_CRYPTO
    print("REG_CRYPTO=", REG_CRYPTO)
except Exception as e:
    print("REG_CRYPTO=ERROR", repr(e))
PY
echo

echo "== Baselines (local) =="
for d in vendor/qlib fork-project/qlib-main fork-project/RD-Agent-main; do
  if [[ -d "$d" ]]; then
    echo "OK: $d"
  else
    echo "MISSING: $d"
  fi
done
echo

echo "== A(vendor/qlib) vs B(fork-project/qlib-main) diff (exclude build artifacts) =="
if [[ -d fork-project/qlib-main/qlib && -d vendor/qlib/qlib ]]; then
  diff -qr \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.so' \
    --exclude='*.cpp' \
    fork-project/qlib-main/qlib vendor/qlib/qlib || true
else
  echo "SKIP: missing fork-project/qlib-main/qlib or vendor/qlib/qlib"
fi
echo

echo "== A(vendor/qlib) vs B(fork-project/qlib-main) key patches =="
if [[ -f fork-project/qlib-main/qlib/__init__.py && -f vendor/qlib/qlib/__init__.py ]]; then
  git diff --no-index -U3 fork-project/qlib-main/qlib/__init__.py vendor/qlib/qlib/__init__.py || true
fi
if [[ -f fork-project/qlib-main/qlib/config.py && -f vendor/qlib/qlib/config.py ]]; then
  git diff --no-index -U3 fork-project/qlib-main/qlib/config.py vendor/qlib/qlib/config.py || true
fi
if [[ -f fork-project/qlib-main/qlib/constant.py && -f vendor/qlib/qlib/constant.py ]]; then
  git diff --no-index -U3 fork-project/qlib-main/qlib/constant.py vendor/qlib/qlib/constant.py || true
fi
if [[ -f fork-project/qlib-main/qlib/utils/time.py && -f vendor/qlib/qlib/utils/time.py ]]; then
  git diff --no-index -U3 fork-project/qlib-main/qlib/utils/time.py vendor/qlib/qlib/utils/time.py || true
fi
if [[ -f fork-project/qlib-main/qlib/data/client.py && -f vendor/qlib/qlib/data/client.py ]]; then
  git diff --no-index -U3 fork-project/qlib-main/qlib/data/client.py vendor/qlib/qlib/data/client.py || true
fi
echo

echo "== Qlib usage (IQFMP) =="
if command -v rg >/dev/null 2>&1; then
  rg -n "\bimport qlib\b|\bfrom qlib\b|qlib\.init\b" -S src/iqfmp | head -n 80 || true
else
  echo "SKIP: rg not installed"
fi
echo

echo "== Qlib usage (RD-Agent local baseline) =="
if command -v rg >/dev/null 2>&1; then
  rg -n "\bimport qlib\b|\bfrom qlib\b|qlib\.init\b|qrun\b" -S fork-project/RD-Agent-main/rdagent fork-project/RD-Agent-main/test | head -n 80 || true
else
  echo "SKIP: rg not installed"
fi
echo

if [[ "$WITH_TESTS" == "--with-tests" ]]; then
  echo "== Tests =="
  pytest -q
fi

