"""Cell, peak and motif axes of the regulatory graph."""

from pathlib import Path
import numpy as np
import scipy.sparse as sp
from .edges import SortedEdgeIndex


def canonical_peaks(names):
    return np.asarray([str(s).replace(":", "_").replace("-", "_") for s in names])


class Dataset:
    def __init__(self, adata, bundle, config):
        self.adata = adata
        self.config = config
        bundle = Path(bundle)
        if not adata.obs_names.is_unique or not adata.var_names.is_unique:
            raise ValueError("Cell and peak IDs must be unique")
        expected = np.asarray((bundle / "peak_names.tsv").read_text().splitlines())
        if not np.array_equal(
            canonical_peaks(adata.var_names), canonical_peaks(expected)
        ):
            raise ValueError(
                "Peak axes differ: input and motif bundle must have identical order"
            )
        self.cell_peak_matrix = sp.csr_matrix(adata.X, copy=True)
        self.cell_peak_matrix.sum_duplicates()
        self.cell_peak_matrix.eliminate_zeros()
        self.cell_peak_matrix.sort_indices()
        if (
            not np.isfinite(self.cell_peak_matrix.data).all()
            or (self.cell_peak_matrix.data < 0).any()
        ):
            raise ValueError("Accessibility must be finite and nonnegative")
        self.n_cells, self.m_peaks = self.cell_peak_matrix.shape
        self.positive_edges = np.column_stack(self.cell_peak_matrix.nonzero()).astype(
            np.int64
        )
        self.positive_counts_by_cell = np.diff(self.cell_peak_matrix.indptr)
        self.edge_index = SortedEdgeIndex.from_pairs(
            self.positive_edges[:, 0], self.positive_edges[:, 1], self.m_peaks
        )
        self.peak_motif_matrix = (
            sp.load_npz(bundle / "peak_motif_train.npz").tocsr().astype(np.float32)
        )
        self.motif_names = np.asarray(
            (bundle / "motif_names.tsv").read_text().splitlines()
        )
        self.n_motifs = len(self.motif_names)
        if self.peak_motif_matrix.shape != (self.m_peaks, self.n_motifs):
            raise ValueError("Peak-motif shape mismatch")
        self.peak_motif_edges = np.column_stack(
            self.peak_motif_matrix.nonzero()
        ).astype(np.int64)
        self.pm_index = SortedEdgeIndex.from_pairs(
            self.peak_motif_edges[:, 0], self.peak_motif_edges[:, 1], self.n_motifs
        )
        # Optional LSI of the cell-by-motif view, used as a second search space for anchors.
        self.motif_lsi_embedding = (
            np.load(bundle / "motif_lsi_cell_svd32.npy")
            if (bundle / "motif_lsi_cell_svd32.npy").exists()
            else None
        )
        self.batch_labels = None

    def lookup_edge_membership(self, cells, peaks):
        return self.edge_index.contains(cells, peaks)

    def lookup_peak_motif_membership(self, peaks, motifs):
        return self.pm_index.contains(peaks, motifs)

    def prepare_alignment(self, parent, batch_key):
        from .anchors import build_batch_anchors

        if batch_key not in self.adata.obs:
            raise ValueError(f"Missing batch column: {batch_key}")
        self.batch_labels = self.adata.obs[batch_key].astype(str).to_numpy()
        self.adata.obsm["cell_embeddings"] = parent
        if self.motif_lsi_embedding is None:
            raise ValueError("Batch alignment requires motif_lsi_cell_svd32.npy in the bundle")
        self.pairs, self.weights, self.sources = build_batch_anchors(
            self, self.batch_labels, k=10
        )
        if len(self.pairs) == 0:
            raise ValueError("No cross-batch anchors found")
        self.base = np.flatnonzero(self.sources == 0)
        self.rescue = np.flatnonzero(self.sources == 1)

    def sample_anchors(self, rng, size=2048):
        # Rescue anchors cover cells without mutual neighbours; they are oversampled so
        # that those cells are also pulled across batches.
        if not len(self.base) or not len(self.rescue):
            ix = rng.choice(len(self.pairs), size=size, replace=len(self.pairs) < size)
        else:
            nr = int(round(size * 0.55))
            nb = size - nr
            ix = np.concatenate(
                [
                    rng.choice(self.base, size=nb, replace=len(self.base) < nb),
                    rng.choice(self.rescue, size=nr, replace=len(self.rescue) < nr),
                ]
            )
            rng.shuffle(ix)
        return self.pairs[ix], self.weights[ix]
