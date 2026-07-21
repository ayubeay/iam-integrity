#!/usr/bin/env bash
# Milestone B build: compile workload.rs -- unchanged -- to wasm32.
#
# No wasm-bindgen, no wasm-pack, no npm. Raw wasm32-unknown-unknown with a
# C-ABI export surface (see src/lib.rs `wasm` module). This keeps the crate
# zero-dependency and keeps framework glue out of the timed region.
set -euo pipefail
cd "$(dirname "$0")"

TARGET=wasm32-unknown-unknown

if ! rustup target list --installed | grep -q "$TARGET"; then
  echo "== adding $TARGET target =="
  rustup target add "$TARGET"
fi

echo "== building lib for $TARGET (release) =="
cargo build --lib --release --target "$TARGET"

SRC="target/$TARGET/release/fair_compute_bench.wasm"
DST="web/fair_compute_bench.wasm"

if [ ! -f "$SRC" ]; then
  echo "error: expected $SRC but it was not produced" >&2
  exit 1
fi

cp "$SRC" "$DST"
SIZE=$(wc -c < "$DST" | tr -d ' ')
echo
echo "wrote $DST ($SIZE bytes)"
echo
echo "The .wasm is a build artifact and is gitignored. Serve the page with:"
echo "    cd web && python3 -m http.server 8000"
echo "then open http://localhost:8000/  (file:// will not load wasm via fetch)."
