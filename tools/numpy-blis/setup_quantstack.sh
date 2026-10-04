#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-.numpy-blis/quantstack-tools}"
mkdir -p "$OUT/pixi"
OUT=$(cd "$OUT" && pwd)
export PIXI_CACHE_DIR="$OUT/cache"
if [ ! -x "$OUT/pixi/pixi" ]; then
  curl -fL https://github.com/prefix-dev/pixi/releases/download/v0.81.0/pixi-x86_64-unknown-linux-musl.tar.gz -o "$OUT/pixi.tar.gz"
  printf '%s  %s\n' 7aa3ec39aecceff9062fa2ed4d42cbaa0bdc25ddea727d048e061cf188d434f6 "$OUT/pixi.tar.gz" | sha256sum -c -
  tar -xzf "$OUT/pixi.tar.gz" -C "$OUT/pixi"
fi
SOURCE="$OUT/source"
PIN=749ef7ac2eb30e4675a718be527dd8c2b00ce1e2
if [ ! -d "$SOURCE" ]; then
  git clone https://github.com/jjerphan/numpy-wasm-openblas-bench.git "$SOURCE"
  git -C "$SOURCE" checkout "$PIN"
fi
[ "$(git -C "$SOURCE" rev-parse HEAD)" = "$PIN" ]
"$OUT/pixi/pixi" install --manifest-path "$SOURCE/pixi.toml" -e default --locked --no-config
for variant in ob034 obdev; do
  "$OUT/pixi/pixi" install --manifest-path "$SOURCE/pixi.toml" -e "$variant" \
    --platform emscripten-wasm32 --locked --no-config
done
printf '%s\n' "$SOURCE"
