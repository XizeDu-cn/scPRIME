"""Build the peak/motif annotation bundle from motifmatchr counts."""

from pathlib import Path
import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from ._core.data import canonical_peaks
from ._core.lsi import tfidf_lsi

MIN_PEAKS_PER_MOTIF = 20
MAX_PEAK_FRACTION_PER_MOTIF = 0.2


def read_scan(bundle, peaks, names):
    """Motif match counts (peaks x motifs) used to weight peaks in TF activity."""
    bundle = Path(bundle)
    if (bundle / "scan_counts.npz").exists():
        pm = sp.load_npz(bundle / "scan_counts.npz").tocsr()
        if pm.shape != (len(peaks), len(names)):
            raise ValueError("Scan count shape mismatch")
        return pm
    with h5py.File(bundle / "freq_motif.h5") as f:
        raw_peaks = f["row_names"].asstr()[()]
        raw_names = f["col_names"].asstr()[()]
        pi = pd.Index(canonical_peaks(raw_peaks)).get_indexer(canonical_peaks(peaks))
        mi = pd.Index(raw_names).get_indexer(names)
        if np.any(np.asarray(pi) < 0) or np.any(mi < 0):
            raise ValueError("Requested scan axes are missing")
        mat = np.asarray(f["mat"][()], np.float32)
        return sp.csr_matrix(mat[np.asarray(pi)][:, mi])


def prepare_bundle(input, scan, output):
    """Filter motifs by frequency and write the bundle consumed by every scprime command."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    a = ad.read_h5ad(input)
    with h5py.File(scan) as f:
        peaks = f["row_names"].asstr()[()]
        names = f["col_names"].asstr()[()]
        mat = np.asarray(f["mat"][()], np.float32)
    if not np.array_equal(canonical_peaks(a.var_names), canonical_peaks(peaks)):
        raise ValueError("Scan peak order must match input")
    binary = mat > 0
    degree = binary.sum(axis=0)
    keep = (degree >= MIN_PEAKS_PER_MOTIF) & (
        degree <= int(np.floor(MAX_PEAK_FRACTION_PER_MOTIF * a.n_vars))
    )
    counts = sp.csr_matrix(mat[:, keep])
    # A random 5% of the matches of each motif are left out of the training edges.
    train = (counts > 0).astype(np.float32).tolil()
    rng = np.random.default_rng(20260328)
    for j in range(train.shape[1]):
        rows = np.flatnonzero(binary[:, np.flatnonzero(keep)[j]])
        held = rng.choice(
            rows,
            size=min(int(np.floor(0.05 * len(rows))), len(rows) - 1),
            replace=False,
        )
        train[held, j] = 0
    train = train.tocsr()
    train.eliminate_zeros()
    if train.shape[1] == 0:
        raise ValueError("No motif passed the frequency filters")
    sp.save_npz(out / "peak_motif_train.npz", train)
    sp.save_npz(out / "scan_counts.npz", counts)
    (out / "motif_names.tsv").write_text("\n".join(names[keep]) + "\n")
    (out / "peak_names.tsv").write_text("\n".join(a.var_names) + "\n")
    np.save(out / "motif_lsi_cell_svd32.npy", tfidf_lsi(sp.csr_matrix(a.X) @ train))
    return out
