"""Cross-batch anchors for scPRIME-BC: prototype-gated mutual nearest neighbours."""

from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from .lsi import tfidf_lsi

# Bit flags recording in which space an anchor pair was found. Pairs supported by both
# the embedding and the TF-IDF LSI space are ranked slightly higher than the others.
EMBEDDING_CANDIDATE = 1
TFIDF_CANDIDATE = 2
MOTIF_CANDIDATE = 4
N_PROTOTYPES = 30
MIN_PROTOTYPE_SIMILARITY = 0.5
UNION_K = 20
PAIRS_PER_CELL = 4
MARGIN_GAP = 0.08
UNION_BONUS = 0.05


def build_batch_anchors(dataset, batch_labels: np.ndarray, k: int):
    """Return anchor pairs (n, 2), their weights (n,) and sources (n,; 0 = mutual neighbour, 1 = rescue).

    Cells are first grouped into prototypes (k-means on the current embedding). Only
    prototypes that match across two batches are searched for mutual nearest neighbours,
    which prevents anchors between unrelated cell populations. Cells left without an
    anchor are rescued with their best partner inside the matched prototypes.
    """
    prototype_emb = np.asarray(dataset.adata.obsm["cell_embeddings"], dtype=np.float32)
    if prototype_emb.ndim != 2 or prototype_emb.shape[0] != dataset.n_cells:
        raise ValueError(
            f"Invalid anchor embedding shape: expected ({dataset.n_cells}, d), got {prototype_emb.shape}"
        )
    prototype_emb = normalize(prototype_emb, norm="l2")
    tfidf_emb = tfidf_lsi(dataset.cell_peak_matrix)
    motif_emb = dataset.motif_lsi_embedding
    cluster_labels = KMeans(
        n_clusters=min(N_PROTOTYPES, dataset.n_cells),
        random_state=int(dataset.config.seed),
        n_init=20,
    ).fit_predict(prototype_emb)
    unique_batches = np.unique(batch_labels)
    centroids = _batch_prototype_centroids(prototype_emb, batch_labels, cluster_labels)
    records = []
    for left_index in range(len(unique_batches)):
        batch_left = unique_batches[left_index]
        cells_left = np.flatnonzero(batch_labels == batch_left)
        for right_index in range(left_index + 1, len(unique_batches)):
            batch_right = unique_batches[right_index]
            cells_right = np.flatnonzero(batch_labels == batch_right)
            if len(cells_left) == 0 or len(cells_right) == 0:
                continue
            prototype_pairs = _match_prototypes(
                centroids.get(str(batch_left), {}), centroids.get(str(batch_right), {})
            )
            pair_map: dict[tuple[int, int], float] = {}
            pair_origin: dict[tuple[int, int], int] = {}
            for cluster_left, cluster_right, _ in prototype_pairs:
                block_left = cells_left[cluster_labels[cells_left] == cluster_left]
                block_right = cells_right[cluster_labels[cells_right] == cluster_right]
                pairs, weights = _mutual_neighbour_pairs(prototype_emb, block_left, block_right, k)
                _merge_pairs(pair_map, pair_origin, pairs, weights, EMBEDDING_CANDIDATE)
                pairs, weights = _mutual_neighbour_pairs(
                    tfidf_emb, block_left, block_right, UNION_K, score_emb=prototype_emb
                )
                _merge_pairs(pair_map, pair_origin, pairs, weights, TFIDF_CANDIDATE)
                if motif_emb is not None:
                    pairs, weights = _mutual_neighbour_pairs(
                        motif_emb, block_left, block_right, UNION_K, score_emb=prototype_emb
                    )
                    _merge_pairs(pair_map, pair_origin, pairs, weights, MOTIF_CANDIDATE)
            pair_map, pair_origin = _limit_pairs_per_cell(pair_map, pair_origin)
            records.append(
                {
                    "cells_left": cells_left,
                    "cells_right": cells_right,
                    "prototype_pairs": prototype_pairs,
                    "pair_map": pair_map,
                    "pair_origin": pair_origin,
                    "rescue_pairs": set(),
                }
            )
    covered_cells: set[int] = set()
    for record in records:
        for cell_left, cell_right in record["pair_map"]:
            covered_cells.add(int(cell_left))
            covered_cells.add(int(cell_right))
    for record in records:
        _rescue_uncovered_cells(record, prototype_emb, cluster_labels, covered_cells)
    anchor_pairs: list[tuple[int, int]] = []
    anchor_weights: list[float] = []
    anchor_sources: list[int] = []
    for record in records:
        pair_map, pair_origin, rescue_pairs = _apply_margin_filter(
            dataset.n_cells, record["pair_map"], record["pair_origin"], record["rescue_pairs"]
        )
        for pair in sorted(pair_map):
            anchor_pairs.append(pair)
            anchor_weights.append(pair_map[pair])
            anchor_sources.append(1 if pair in rescue_pairs else 0)
    return (
        np.asarray(anchor_pairs, dtype=np.int64),
        np.asarray(anchor_weights, dtype=np.float32),
        np.asarray(anchor_sources, dtype=np.int8),
    )


def _batch_prototype_centroids(
    prototype_emb: np.ndarray, batch_labels: np.ndarray, cluster_labels: np.ndarray
) -> dict[str, dict[int, np.ndarray]]:
    centroids: dict[str, dict[int, np.ndarray]] = {}
    for batch in np.unique(batch_labels):
        cells = np.flatnonzero(batch_labels == batch)
        batch_centroids: dict[int, np.ndarray] = {}
        for cluster_id in np.unique(cluster_labels[cells]):
            cluster_cells = cells[cluster_labels[cells] == cluster_id]
            if len(cluster_cells) == 0:
                continue
            center = prototype_emb[cluster_cells].mean(axis=0, keepdims=True)
            batch_centroids[int(cluster_id)] = normalize(center, norm="l2")[0]
        centroids[str(batch)] = batch_centroids
    return centroids


def _match_prototypes(
    centroids_left: dict[int, np.ndarray], centroids_right: dict[int, np.ndarray]
) -> list[tuple[int, int, float]]:
    """Prototypes of two batches are matched when they are mutual top-2 neighbours."""
    if not centroids_left or not centroids_right:
        return []
    left_ids = sorted(centroids_left)
    right_ids = sorted(centroids_right)
    left_matrix = np.vstack([centroids_left[idx] for idx in left_ids])
    right_matrix = np.vstack([centroids_right[idx] for idx in right_ids])
    similarity = np.clip(left_matrix @ right_matrix.T, -1.0, 1.0)
    k_right_per_left = min(2, similarity.shape[1])
    k_left_per_right = min(2, similarity.shape[0])
    left_topk_right = np.argsort(-similarity, axis=1)[:, :k_right_per_left]
    right_topk_left = np.argsort(-similarity.T, axis=1)[:, :k_left_per_right]
    matched_pairs = []
    for local_left, right_neighbors in enumerate(left_topk_right):
        for local_right in right_neighbors:
            sim = float(similarity[local_left, local_right])
            if sim < MIN_PROTOTYPE_SIMILARITY:
                continue
            if local_left not in right_topk_left[int(local_right)]:
                continue
            matched_pairs.append((int(left_ids[local_left]), int(right_ids[int(local_right)]), sim))
    return matched_pairs


def _mutual_neighbour_pairs(
    search_emb: np.ndarray,
    cells_left: np.ndarray,
    cells_right: np.ndarray,
    k: int,
    score_emb: np.ndarray | None = None,
) -> tuple[list[tuple[int, int]], list[float]]:
    """Mutual k-nearest neighbours between two cell blocks, weighted by cosine similarity.

    Neighbours are searched in ``search_emb``; when ``score_emb`` is given the weight is
    the cosine similarity in that space instead (used for the LSI candidate spaces).
    """
    k_right_per_left = min(int(k), len(cells_right))
    k_left_per_right = min(int(k), len(cells_left))
    if k_right_per_left == 0 or k_left_per_right == 0:
        return ([], [])
    nn_for_left = NearestNeighbors(metric="cosine", n_neighbors=k_right_per_left)
    nn_for_left.fit(search_emb[cells_right])
    dist_left_to_right, left_topk_right = nn_for_left.kneighbors(search_emb[cells_left])
    nn_for_right = NearestNeighbors(metric="cosine", n_neighbors=k_left_per_right)
    nn_for_right.fit(search_emb[cells_left])
    _, right_topk_left = nn_for_right.kneighbors(search_emb[cells_right])
    right_neighbor_sets = [set(neighbors.tolist()) for neighbors in right_topk_left]
    block_pairs = []
    block_weights = []
    for local_left, right_neighbors in enumerate(left_topk_right):
        global_left = int(cells_left[local_left])
        for neighbor_rank, local_right in enumerate(right_neighbors):
            if local_left not in right_neighbor_sets[int(local_right)]:
                continue
            global_right = int(cells_right[int(local_right)])
            if score_emb is None:
                cosine_sim = max(1.0 - float(dist_left_to_right[local_left, neighbor_rank]), 1e-06)
            else:
                cosine_sim = max(float(score_emb[global_left] @ score_emb[global_right]), 1e-06)
            block_pairs.append((global_left, global_right))
            block_weights.append(cosine_sim)
    return (block_pairs, block_weights)


def _merge_pairs(
    pair_map: dict[tuple[int, int], float],
    pair_origin: dict[tuple[int, int], int],
    pairs: list[tuple[int, int]],
    weights: list[float],
    origin: int,
) -> None:
    # Keep the largest weight seen for a pair; the origin flags accumulate across the
    # pairs merged in one call.
    for pair, weight in zip(pairs, weights):
        existing_origin = pair_origin.get(pair)
        if existing_origin is not None:
            origin |= int(existing_origin)
        if pair not in pair_map or weight > pair_map[pair]:
            pair_map[pair] = float(weight)
            pair_origin[pair] = int(origin)
        elif existing_origin is not None and int(existing_origin) != int(origin):
            pair_origin[pair] = int(origin)


def _rank_scores(
    ordered_pairs: list[tuple[int, int]],
    pair_map: dict[tuple[int, int], float],
    pair_origin: dict[tuple[int, int], int],
) -> tuple[np.ndarray, np.ndarray]:
    raw_weights = np.asarray([pair_map[pair] for pair in ordered_pairs], dtype=np.float32)
    origins = np.asarray([pair_origin[pair] for pair in ordered_pairs], dtype=np.int8)
    rank_scores = raw_weights.astype(np.float32, copy=True)
    rank_scores[origins == EMBEDDING_CANDIDATE | TFIDF_CANDIDATE] += float(UNION_BONUS)
    return (rank_scores, raw_weights)


def _limit_pairs_per_cell(
    pair_map: dict[tuple[int, int], float], pair_origin: dict[tuple[int, int], int]
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], int]]:
    """Keep at most PAIRS_PER_CELL anchors per cell, best-ranked first."""
    if not pair_map:
        return (pair_map, pair_origin)
    ordered_pairs = list(pair_map.keys())
    rank_scores, raw_weights = _rank_scores(ordered_pairs, pair_map, pair_origin)
    ranked_pairs = sorted(
        zip(ordered_pairs, rank_scores.tolist(), raw_weights.tolist()),
        key=lambda item: (-item[1], -item[2], item[0][0], item[0][1]),
    )
    kept_pairs: dict[tuple[int, int], float] = {}
    kept_origin: dict[tuple[int, int], int] = {}
    used_left_counts: dict[int, int] = {}
    used_right_counts: dict[int, int] = {}
    for (cell_left, cell_right), _, weight in ranked_pairs:
        left_full = used_left_counts.get(int(cell_left), 0) >= PAIRS_PER_CELL
        right_full = used_right_counts.get(int(cell_right), 0) >= PAIRS_PER_CELL
        if left_full or right_full:
            continue
        kept_pairs[cell_left, cell_right] = weight
        kept_origin[cell_left, cell_right] = pair_origin[cell_left, cell_right]
        used_left_counts[int(cell_left)] = used_left_counts.get(int(cell_left), 0) + 1
        used_right_counts[int(cell_right)] = used_right_counts.get(int(cell_right), 0) + 1
    return (kept_pairs, kept_origin)


def _rescue_uncovered_cells(
    record: dict, prototype_emb: np.ndarray, cluster_labels: np.ndarray, covered_cells: set[int]
) -> None:
    """Give every still-unanchored cell its most similar partner inside the matched prototypes."""
    prototype_pairs = record["prototype_pairs"]
    if not prototype_pairs:
        return
    pair_map = record["pair_map"]
    pair_origin = record["pair_origin"]
    rescue_pairs = record["rescue_pairs"]
    left_to_right_clusters: dict[int, set[int]] = {}
    right_to_left_clusters: dict[int, set[int]] = {}
    for cluster_left, cluster_right, _ in prototype_pairs:
        left_to_right_clusters.setdefault(int(cluster_left), set()).add(int(cluster_right))
        right_to_left_clusters.setdefault(int(cluster_right), set()).add(int(cluster_left))
    left_cells_by_cluster: dict[int, list[int]] = {}
    right_cells_by_cluster: dict[int, list[int]] = {}
    for cell in record["cells_left"]:
        left_cells_by_cluster.setdefault(int(cluster_labels[int(cell)]), []).append(int(cell))
    for cell in record["cells_right"]:
        right_cells_by_cluster.setdefault(int(cluster_labels[int(cell)]), []).append(int(cell))
    for cell_left in record["cells_left"]:
        cell_left = int(cell_left)
        if cell_left in covered_cells:
            continue
        partner = _best_rescue_partner(
            cell_left,
            prototype_emb,
            left_to_right_clusters.get(int(cluster_labels[cell_left]), set()),
            right_cells_by_cluster,
        )
        if partner is None:
            continue
        partner_cell, similarity = partner
        pair = (cell_left, partner_cell)
        if pair not in pair_map or similarity > pair_map[pair]:
            if pair not in pair_map:
                pair_origin[pair] = EMBEDDING_CANDIDATE
                rescue_pairs.add(pair)
            pair_map[pair] = similarity
            covered_cells.add(cell_left)
            covered_cells.add(partner_cell)
    for cell_right in record["cells_right"]:
        cell_right = int(cell_right)
        if cell_right in covered_cells:
            continue
        partner = _best_rescue_partner(
            cell_right,
            prototype_emb,
            right_to_left_clusters.get(int(cluster_labels[cell_right]), set()),
            left_cells_by_cluster,
        )
        if partner is None:
            continue
        partner_cell, similarity = partner
        pair = (partner_cell, cell_right)
        if pair not in pair_map or similarity > pair_map[pair]:
            if pair not in pair_map:
                pair_origin[pair] = EMBEDDING_CANDIDATE
                rescue_pairs.add(pair)
            pair_map[pair] = similarity
            covered_cells.add(partner_cell)
            covered_cells.add(cell_right)


def _best_rescue_partner(
    source_cell: int,
    prototype_emb: np.ndarray,
    allowed_target_clusters: set[int],
    target_cells_by_cluster: dict[int, list[int]],
) -> tuple[int, float] | None:
    if not allowed_target_clusters:
        return None
    candidate_cells = []
    for cluster_id in allowed_target_clusters:
        candidate_cells.extend(target_cells_by_cluster.get(int(cluster_id), []))
    if not candidate_cells:
        return None
    candidate_cells_array = np.asarray(candidate_cells, dtype=np.int64)
    similarities = prototype_emb[candidate_cells_array] @ prototype_emb[int(source_cell)]
    best_index = int(np.argmax(similarities))
    best_similarity = float(similarities[best_index])
    if best_similarity < MIN_PROTOTYPE_SIMILARITY:
        return None
    return (int(candidate_cells_array[best_index]), best_similarity)


def _apply_margin_filter(
    n_cells: int,
    pair_map: dict[tuple[int, int], float],
    pair_origin: dict[tuple[int, int], int],
    rescue_pairs: set[tuple[int, int]],
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], int], set[tuple[int, int]]]:
    """Drop anchors that score clearly below the best anchor of either of their cells."""
    if not pair_map:
        return (pair_map, pair_origin, rescue_pairs)
    ordered_pairs = list(pair_map.keys())
    left_cells = np.asarray([pair[0] for pair in ordered_pairs], dtype=np.int64)
    right_cells = np.asarray([pair[1] for pair in ordered_pairs], dtype=np.int64)
    rank_scores, weights = _rank_scores(ordered_pairs, pair_map, pair_origin)
    best_left = _best_value_by_cell(n_cells, left_cells, rank_scores)
    best_right = _best_value_by_cell(n_cells, right_cells, rank_scores)
    margin_gap = np.maximum(best_left[left_cells] - rank_scores, best_right[right_cells] - rank_scores)
    keep_mask = margin_gap <= float(MARGIN_GAP)
    filtered_pairs: dict[tuple[int, int], float] = {}
    filtered_origin: dict[tuple[int, int], int] = {}
    filtered_rescue_pairs: set[tuple[int, int]] = set()
    for index, pair in enumerate(ordered_pairs):
        if not bool(keep_mask[index]):
            continue
        filtered_pairs[pair] = float(weights[index])
        filtered_origin[pair] = pair_origin[pair]
        if pair in rescue_pairs:
            filtered_rescue_pairs.add(pair)
    return (filtered_pairs, filtered_origin, filtered_rescue_pairs)


def _best_value_by_cell(n_cells: int, cell_ids: np.ndarray, values: np.ndarray) -> np.ndarray:
    best = np.full(n_cells, -np.inf, dtype=np.float32)
    np.maximum.at(best, cell_ids.astype(np.int64, copy=False), values.astype(np.float32, copy=False))
    return best
