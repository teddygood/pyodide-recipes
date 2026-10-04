#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT="${1:-$ROOT/.numpy-blis}"
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

python -c 'from pyodide_build.build_env import init_environment, get_build_flag; init_environment(); assert get_build_flag("PYODIDE_VERSION") == "315.0.0a2"'
[ "$(pyodide config get pyodide_abi_version)" = "2026_5" ]
emcc --version | head -1 | grep -F '6.0.5'

pyodide build-recipes numpy \
  --recipe-dir="$ROOT/tools/numpy-blis/recipes" \
  --build-dir="$OUT/build" --install --install-dir="$OUT/dist" \
  --log-dir="$OUT/logs" --n-jobs=1

RUNTIME=$(pyodide config get dist_dir)
for file in pyodide.js pyodide.mjs pyodide.asm.mjs pyodide.asm.wasm python_stdlib.zip; do
  cp "$RUNTIME/$file" "$OUT/dist/$file"
done
emcc "$ROOT/tools/numpy-blis/dispatch_probe.c" \
  -I"$OUT/build/.libs/include/blis" -sSIDE_MODULE=1 -O2 \
  -o "$OUT/dispatch_probe.so"

python "$ROOT/tools/numpy-blis/inspect_wheel.py" "$OUT"
