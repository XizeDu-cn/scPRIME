#!/usr/bin/env bash
# scPRIME-BC on the hematopoiesis dataset: embeddings are trained for 10 epochs and then
# aligned across donors for 10 epochs.
set -euo pipefail
data=${1:?Usage: bash run_batch_alignment.sh DATA_DIRECTORY OUTPUT_DIRECTORY}
out=${2:?Output directory required}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export CUBLAS_WORKSPACE_CONFIG=:4096:8

scprime train --input "$data/input.h5ad" --bundle "$data/motif_bundle" --output "$out/embedding" --epochs 10
scprime align-batches --input "$data/input.h5ad" --bundle "$data/motif_bundle" --parent "$out/embedding" --output "$out/batch_aligned" --batch-key batch --epochs 10
