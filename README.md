# scPRIME

scPRIME (single-cell Path-based Regulatory Inference and Multi-entity Embedding)
models scATAC-seq data as a regulatory graph of cells, peaks and TF motifs. From a
cell-by-peak matrix and motif annotations of the peaks it produces

- cell embeddings for visualization, clustering and batch correction (scPRIME-BC),
- the TF regulator families of each cell type together with the peaks that support them,
- a single-cell TF activity matrix.

Manuscript: *scPRIME: path-based regulatory inference and multi-entity embedding for
single-cell chromatin accessibility* (preprint forthcoming).

## Installation

Python 3.12 and PyTorch 2.3.0 are the reference versions. With an NVIDIA GPU:

```bash
conda env create -f environment.yml
conda activate scprime
pip install -e .
```

CPU-only installation, followed by the bundled minimal example (300 cells, a few minutes):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e .
bash examples/run_minimal_example.sh outputs/minimal
```

## Quick start: human hematopoiesis

The example data (Buenrostro et al., 2018; GSE96769; 2,034 cells) are distributed as a
checksum-verified archive, DOI to follow. Download, run the pipeline, and optionally
the batch-corrected variant:

```bash
bash examples/buenrostro/download_data.sh <archive URL>
bash examples/buenrostro/run_pipeline.sh examples/buenrostro/data outputs/buenrostro
bash examples/buenrostro/run_batch_alignment.sh examples/buenrostro/data outputs/buenrostro_bc
```

`run_pipeline.sh` runs four commands:

| Command | What it does | Output |
|---|---|---|
| `scprime train` | Learns cell, peak and motif embeddings (20 epochs) | `embedding/` |
| `scprime calibrate` | Learns the relation-specific calibration on the fixed embeddings (8 epochs) | `calibration/` |
| `scprime infer-regulators` | Ranks motif families for each cell type and lists supporting peaks | `regulators/` |
| `scprime tf-activity` | Computes single-cell TF activity | `tf_activity.h5ad` |

`run_batch_alignment.sh` trains the embedding and then runs `scprime align-batches`,
which continues training with cross-batch anchors (`batch_aligned/`). Outputs of both
scripts on the full dataset are provided in `examples/buenrostro/expected_results/`. Every command
takes `--seed` (default 15), `--epochs` and `--device cpu|cuda`; run
`scprime <command> --help` for the arguments.

## Inputs

- `input.h5ad`: cells x peaks. `X` holds nonnegative accessibility (binary recommended).
  Cell and peak IDs must be unique. `obs['celltype']` is needed only by
  `infer-regulators` (`--label-key` selects another column, `--exclude-labels` skips
  labels such as `UNK`); `obs['batch']` is needed only by `align-batches` (`--batch-key`).
- `motif_bundle/`: created by `scprime prepare-motifs` from motifmatchr counts.
  - `peak_motif_train.npz`: binary peak x motif matrix defining the peak–motif edges
  - `scan_counts.npz`: motif match counts used to weight peaks in TF activity
  - `peak_names.tsv`, `motif_names.tsv`: one ID per line, in matrix order
  - `motif_lsi_cell_svd32.npy`: LSI of the cell-by-motif view, used when searching anchors
- `family_map.tsv`: two columns, `motif_name` and `family`, grouping motifs into the
  families ranked by `infer-regulators`. A family without any motif in the vocabulary
  receives no score and is listed last.

To annotate a new dataset, extract the peak sequences and run the motif scan:

```bash
bedtools getfasta -fi genome.fa -bed peaks.bed -fo peaks.fa
Rscript scripts/scan_motifs.R --fasta peaks.fa --output motif_counts.h5
scprime prepare-motifs --input input.h5ad --scan motif_counts.h5 --output motif_bundle
```

The R script needs the Bioconductor packages `JASPAR2020`, `TFBSTools`, `motifmatchr`,
`Biostrings`, `SummarizedExperiment`, `HDF5Array` and `rhdf5`, plus `optparse`. Peak
names in the FASTA and the h5ad must match (`chr:start-end` and `chr_start_end` are
treated as equal). Motifs matching fewer than 20 peaks or more than 20% of peaks are
excluded, which leaves 609 motifs on the hematopoiesis dataset.

## Outputs

| File | Contents |
|---|---|
| `embedding/embedding.h5ad` | `obsm['cell_embeddings']` (cells x 50) and `varm['peak_embeddings']` (peaks x 50); `X` is an empty placeholder |
| `embedding/motif_embeddings.npy` | motifs x 50, in the order of `motif_names.tsv` |
| `calibration/embedding.h5ad` | as above, plus the projections and relation angles in `uns` |
| `regulators/family_rankings.tsv` | cell type, family, rank and score |
| `regulators/supporting_peaks.tsv` | the 50 top-scoring supporting peaks per cell type and family, with their score components |
| `tf_activity.h5ad` | cells x motifs activity `M(c, m)`; standardize per motif for visualization |
| `batch_aligned/embedding.h5ad` | batch-aligned cell embeddings (scPRIME-BC) |

## Default hyperparameters

| Symbol | Meaning | Default |
|---|---|---|
| d | Embedding dimension | 50 |
| λ_PM | Weight of the peak–motif loss | 0.1 |
| K, n | Candidate and sampled negative peaks per cell | 200, 50 |
| α | Temperature of adaptive negative sampling | 3 |
| — | Embedding epochs, learning rate (Adagrad), weight decay | 20, 0.1, 0.00273 |
| d′, γ | Dimension and margin of the relation calibration | 24, 12 |
| — | Calibration epochs, negatives per edge, peak–motif edges per epoch | 8, 32, 10% |
| N_s | Specific and background peaks per regulator query | 3,000 |

Batch alignment (scPRIME-BC) uses mutual-nearest-neighbour anchors between batches,
restricted to matching cell prototypes; details are given in the Supplementary Material.

## Citation and license

Citation: preprint forthcoming. Code is released under the MIT license. The example
data derive from GSE96769 (Buenrostro et al., *Cell*, 2018) and remain subject to the
original data terms.
