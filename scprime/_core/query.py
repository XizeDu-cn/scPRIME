"""Path-based scoring of motifs for a cell type (Eq. 4 of the manuscript)."""

from __future__ import annotations
from typing import Any, Mapping, Sequence
import numpy as np
from scipy.special import logsumexp

N_SPECIFIC_PEAKS = 1500


def zscore(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    std = float(np.nanstd(array))
    if std < 1e-12:
        return np.zeros_like(array, dtype=np.float64)
    return (array - float(np.nanmean(array))) / std


def rank_z_rows(values: np.ndarray) -> np.ndarray:
    """Rank-standardize each row: ranks mapped to [0, 1], then centred and scaled."""
    raw = np.asarray(values, dtype=np.float32)
    n_rows, n_columns = raw.shape
    order = np.argsort(raw, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    scale = np.linspace(0.0, 1.0, n_columns, dtype=np.float32)
    ranks[np.arange(n_rows)[:, None], order] = scale[None, :]
    std = float(np.std(scale))
    if std < 1e-08:
        return np.zeros_like(ranks, dtype=np.float32)
    return ((ranks - float(np.mean(scale))) / std).astype(np.float32, copy=False)


def embedding_scores(arrays: Mapping[str, Any], peak_ids: np.ndarray) -> np.ndarray:
    """Inner-product score of peak and motif embeddings for peaks x motifs."""
    peaks = np.asarray(arrays["peak_embeddings"], dtype=np.float32)[peak_ids]
    motifs = np.asarray(arrays["motif_embeddings"], dtype=np.float32)
    return np.asarray(peaks @ motifs.T, dtype=np.float32)


def path_scores(
    peak_stats: Mapping[str, Mapping[str, np.ndarray]],
    arrays: Mapping[str, Any],
    family_ids: Mapping[str, np.ndarray],
    cell_types: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Family scores per cell type from cell-type -> peak -> motif paths.

    The most specific peaks of a cell type are aggregated with a log-sum-exp over
    specificity plus the rank-standardized peak-motif embedding score; the least
    specific peaks serve as background and are subtracted. A family takes the maximum
    over its motifs.
    """
    output: dict[str, dict[str, float]] = {}
    for cell_type in cell_types:
        specificity = np.asarray(peak_stats[cell_type]["specificity"], dtype=np.float32)
        top = np.argsort(-specificity, kind="mergesort")[:N_SPECIFIC_PEAKS].astype(np.int64)
        background = np.argsort(specificity, kind="mergesort")[:N_SPECIFIC_PEAKS].astype(np.int64)
        top_embedding = rank_z_rows(embedding_scores(arrays, top))
        background_embedding = rank_z_rows(embedding_scores(arrays, background))
        motif_scores = logsumexp(specificity[top, None] + top_embedding, axis=0)
        motif_scores -= 0.5 * logsumexp(-specificity[background, None] + background_embedding, axis=0)
        output[cell_type] = {
            family: float(np.max(motif_scores[ids])) if len(ids) else float("nan")
            for family, ids in family_ids.items()
        }
    return output
