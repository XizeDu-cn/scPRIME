#!/usr/bin/env bash
# Runs every scprime command on the bundled 300-cell subset. Finishes in a few minutes on CPU.
set -euo pipefail
out=${1:?Usage: bash examples/run_minimal_example.sh OUTPUT_DIRECTORY}
data=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/minimal_data" && pwd)
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8

scprime train --input "$data/input.h5ad" --bundle "$data/motif_bundle" --output "$out/embedding" --epochs 2 --device cpu
scprime infer-regulators --input "$data/input.h5ad" --bundle "$data/motif_bundle" --model "$out/embedding" --family-map "$data/family_map.tsv" --exclude-labels UNK --output "$out/regulators"
scprime tf-activity --input "$data/input.h5ad" --bundle "$data/motif_bundle" --model "$out/embedding" --output "$out/tf_activity.h5ad"
scprime align-batches --input "$data/input.h5ad" --bundle "$data/motif_bundle" --parent "$out/embedding" --output "$out/batch_aligned" --batch-key batch --epochs 1 --device cpu
