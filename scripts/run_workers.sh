#!/usr/bin/env bash
# Run the builder as N CPU workers, each on its own slice of the grid and its own warp kernel
# cache (two processes sharing one cache corrupt it). Resumable: rerun and finished samples
# are skipped. Usage: scripts/run_workers.sh <config> <out_dir> [n_workers]
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
CFG="${1:?config yaml}"; OUT="${2:?output dir}"; N="${3:-4}"
PY="${PHOTOAI_PYTHON:-$HERE/../fitforgesolver/.venv/bin/python}"
mkdir -p "$OUT/logs"
for ((i=0; i<N; i++)); do
  WARP_CACHE_PATH="$HERE/.warp-cache-w$i" CUDA_VISIBLE_DEVICES=-1 \
    nohup "$PY" -m photoai run --config "$CFG" --out "$OUT" --shard "$i" --n-shards "$N" \
    > "$OUT/logs/worker-$i.log" 2>&1 &
  echo "worker $i pid $!"
done
wait
