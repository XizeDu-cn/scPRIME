#!/usr/bin/env bash
# Full scPRIME pipeline on the human hematopoiesis dataset (Buenrostro et al., 2018).
set -euo pipefail
data=${1:?Usage: bash run_pipeline.sh DATA_DIRECTORY OUTPUT_DIRECTORY}
out=${2:?Output directory required}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export CUBLAS_WORKSPACE_CONFIG=:4096:8

scprime train --input "$data/input.h5ad" --bundle "$data/motif_bundle" --output "$out/embedding"
scprime calibrate --input "$data/input.h5ad" --bundle "$data/motif_bundle" --parent "$out/embedding" --output "$out/calibration"
scprime infer-regulators --input "$data/input.h5ad" --bundle "$data/motif_bundle" --model "$out/calibration" --family-map "$data/family_map.tsv" --exclude-labels UNK --output "$out/regulators"
scprime tf-activity --input "$data/input.h5ad" --bundle "$data/motif_bundle" --model "$out/embedding" --output "$out/tf_activity.h5ad"
