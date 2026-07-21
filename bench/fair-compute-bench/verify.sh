#!/usr/bin/env bash
# Cross-check the Rust implementation against the independent Python reference.
# This is the gate for Milestone A. If it fails, the workload is not specified
# well enough to carry into Milestone B.
set -euo pipefail
cd "$(dirname "$0")"

echo "== structural check on the access pattern =="
python3 reference/access_pattern_check.py

echo
echo "== building (release) =="
cargo build --release --quiet

echo
echo "== unit + KAT tests =="
cargo test --release --quiet

echo
echo "== rust known-answer vectors =="
./target/release/fair-compute-bench --print-kat | tee /tmp/kat_rust.txt

echo
echo "== python reference vectors =="
python3 reference/reference_workload.py | grep '^KAT' | tee /tmp/kat_py.txt

echo
if diff -u /tmp/kat_py.txt /tmp/kat_rust.txt; then
  echo "MATCH: rust and python agree on all vectors."
else
  echo "MISMATCH: implementations disagree. Milestone A is not complete."
  exit 1
fi
