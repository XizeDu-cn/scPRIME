# scPRIME tutorial

After installing the package, run the bundled minimal example:

```bash
bash examples/run_minimal_example.sh outputs/tutorial
```

It trains on a 300-cell subset of the hematopoiesis dataset with shortened epochs, so it
exercises every command but does not reproduce the numbers in the manuscript. For the
full dataset use the scripts in `examples/buenrostro/` (see the README).

The same steps from Python:

```python
from scprime import train, calibrate, infer_regulators, tf_activity, align_batches

train("data/input.h5ad", "data/motif_bundle", "outputs/embedding")
calibrate("data/input.h5ad", "data/motif_bundle", "outputs/embedding", "outputs/calibration")
infer_regulators("data/input.h5ad", "data/motif_bundle", "outputs/calibration",
                 "data/family_map.tsv", "outputs/regulators", exclude_labels=("UNK",))
tf_activity("data/input.h5ad", "data/motif_bundle", "outputs/embedding", "outputs/tf_activity.h5ad")
align_batches("data/input.h5ad", "data/motif_bundle", "outputs/embedding",
              "outputs/batch_aligned", batch_key="batch")
```

Reading the outputs:

```python
import anndata as ad, numpy as np, pandas as pd

emb = ad.read_h5ad("outputs/embedding/embedding.h5ad")
cells = emb.obsm["cell_embeddings"]          # cells x 50, cluster or embed with UMAP
peaks = emb.varm["peak_embeddings"]          # peaks x 50
motifs = np.load("outputs/embedding/motif_embeddings.npy")

rankings = pd.read_csv("outputs/regulators/family_rankings.tsv", sep="\t")
rankings[rankings.cell_type == "MEP"].head()  # GATA should rank first

activity = ad.read_h5ad("outputs/tf_activity.h5ad")   # cells x motifs
```

TF activity is the raw score `M(c, m)`; standardize each motif across cells before
plotting. Supporting peaks are the peaks that contribute most to a cell-type–family
query, listed with their specificity and peak–motif score components.
