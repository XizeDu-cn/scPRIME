#!/usr/bin/env bash
# Fetch and verify the hematopoiesis example data (input.h5ad, motif_bundle/, family_map.tsv).
set -euo pipefail
source=${1:?Usage: bash download_data.sh URL_OR_LOCAL_ARCHIVE [DESTINATION]}
dest=${2:-examples/buenrostro/data}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
expected=$(cut -d ' ' -f1 "$script_dir/data_archive.sha256")
temporary=$(mktemp -d)
trap 'rm -f "$temporary/data.tar.gz"; rmdir "$temporary"' EXIT
case "$source" in
  https://*|http://*) curl --fail --location --retry 3 "$source" --output "$temporary/data.tar.gz" ;;
  *) cp -- "$source" "$temporary/data.tar.gz" ;;
esac
printf '%s  %s\n' "$expected" "$temporary/data.tar.gz" | sha256sum --check --status
mkdir -p "$dest"
if [ -n "$(ls -A "$dest")" ]; then echo 'Destination must be empty' >&2; exit 1; fi
tar -xzf "$temporary/data.tar.gz" -C "$dest"
