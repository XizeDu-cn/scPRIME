"""Training and inference entry points of scPRIME."""

from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from ._core.model import EmbeddingModel
from ._core.data import Dataset
from ._core.random import Streams, seed_all
from ._core.query import (
    N_SPECIFIC_PEAKS,
    path_scores,
    relation_scores,
    embedding_scores,
    rank_z_rows,
    zscore,
)

# Fraction of peak-motif edges revisited in each calibration epoch.
CALIBRATION_PM_FRACTION = 0.1
# Weight of the anchor term during batch alignment.
ALIGNMENT_WEIGHT = 800.0


@dataclass
class TrainConfig:
    embedding_dim: int = 50
    epochs: int = 20
    seed: int = 15
    lr: float = 0.1
    wd: float = 0.00273
    wd_interval: int = 50
    batch_size: int = 60480
    num_batch_negs: int = 64
    num_uniform_negs: int = 50
    pm_loss_weight: float = 0.1
    adaptive_negatives: bool = True
    adv_pool_size: int = 200
    adv_alpha: float = 3.0


def _read_model(path):
    return ad.read_h5ad(Path(path) / "embedding.h5ad"), np.load(
        Path(path) / "motif_embeddings.npy"
    )


def _train(
    input,
    bundle,
    output,
    config,
    parent=None,
    batch_key=None,
    calibration=False,
    device=None,
):
    started = time.perf_counter()
    if config.epochs < 1 or config.seed < 0:
        raise ValueError("epochs must be positive and seed nonnegative")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_all(config.seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    a = ad.read_h5ad(input)
    dataset = Dataset(a, bundle, config)
    if not len(dataset.positive_edges) or not len(dataset.peak_motif_edges):
        raise ValueError(
            "Training requires nonempty accessibility and peak-motif edges"
        )
    model = EmbeddingModel(dataset, config, calibration).to(device)
    streams = Streams(config.seed, device)
    model.rng_streams = streams
    if parent:
        p, mm = _read_model(parent)
        if not a.obs_names.equals(p.obs_names) or not a.var_names.equals(p.var_names):
            raise ValueError("Warm-start cell/peak axes differ")
        with torch.no_grad():
            model.cell_embedding.weight.copy_(
                torch.as_tensor(p.obsm["cell_embeddings"], device=device)
            )
            model.peak_embedding.weight.copy_(
                torch.as_tensor(p.varm["peak_embeddings"], device=device)
            )
            model.motif_embedding.weight.copy_(torch.as_tensor(mm, device=device))
        if calibration:
            # Calibration learns only the projections and relations; the embeddings stay fixed.
            for layer in [
                model.cell_embedding,
                model.peak_embedding,
                model.motif_embedding,
            ]:
                layer.weight.requires_grad_(False)
        elif batch_key:
            dataset.prepare_alignment(np.asarray(p.obsm["cell_embeddings"]), batch_key)
    if calibration:
        dataset.prepare_cell_motif()
    # Embedding tables use sparse gradients and form their own parameter group.
    dense = []
    sparse = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            (
                sparse
                if name.startswith(
                    ("cell_embedding.", "peak_embedding.", "motif_embedding.")
                )
                else dense
            ).append(param)
    groups = [{"params": ps, "lr": config.lr} for ps in [dense, sparse] if ps]
    optimizer = torch.optim.Adagrad(groups, lr=config.lr)
    edges = dataset.positive_edges
    n = len(edges)
    bs = config.batch_size
    npm = len(dataset.peak_motif_edges)
    with (out / "training.jsonl").open("w") as log:
        for epoch in range(config.epochs):
            tick = time.perf_counter()
            # Draw the per-cell negative order first so that the edge permutation below
            # does not depend on how many cells have edges.
            for count in dataset.positive_counts_by_cell:
                if count > 0:
                    streams.order.choice(
                        dataset.m_peaks - int(count), size=int(count), replace=True
                    )
            order = streams.perm.permutation(n)
            if calibration:
                po = streams.pm.permutation(npm)[: max(1, int(npm * CALIBRATION_PM_FRACTION))]
                cm = dataset.cell_motif_edges[
                    streams.pm.permutation(len(dataset.cell_motif_edges))
                ]
            elif config.pm_loss_weight > 0:
                po = streams.perm.permutation(npm)
            else:
                po = None
            pm_bs = min(bs, len(po)) if po is not None else 0
            cm_bs = min(bs, len(cm)) if calibration else 0
            total = 0.0
            for b, start in enumerate(range(0, n, bs)):
                optimizer.zero_grad(set_to_none=True)
                cp = torch.as_tensor(
                    edges[order[start : start + bs]], device=device, dtype=torch.long
                )
                loss = model.cell_peak_loss(
                    cp[:, 0],
                    cp[:, 1],
                    num_batch_negs=config.num_batch_negs,
                    num_uniform_negs=config.num_uniform_negs,
                )
                pm = None
                if po is not None:
                    ix = po[(np.arange(pm_bs) + b * pm_bs) % len(po)]
                    pm = torch.as_tensor(
                        dataset.peak_motif_edges[ix], device=device, dtype=torch.long
                    )
                    if config.pm_loss_weight > 0:
                        loss = loss + config.pm_loss_weight * model.peak_motif_loss(
                            pm[:, 0], pm[:, 1]
                        )
                if calibration:
                    cc = torch.as_tensor(
                        cm[(np.arange(cm_bs) + b * cm_bs) % len(cm)],
                        device=device,
                        dtype=torch.long,
                    )
                    # The composition penalty is ramped in over the first two epochs.
                    model.composition_scale = min(1.0, epoch / 2.0)
                    loss = loss + model.relation_loss(
                        cp[:, 0], cp[:, 1], pm[:, 0], pm[:, 1], cc[:, 0], cc[:, 1]
                    )
                if config.wd > 0 and streams.python.random() < 1.0 / config.wd_interval:
                    loss = loss + config.wd * config.wd_interval * model.l2_norm()
                if batch_key:
                    pairs, weights = dataset.sample_anchors(streams.anchor)
                    pairs = torch.as_tensor(pairs, device=device, dtype=torch.long)
                    loss = loss + ALIGNMENT_WEIGHT * model.alignment_loss(
                        pairs[:, 0],
                        pairs[:, 1],
                        torch.as_tensor(weights, device=device, dtype=torch.float32),
                    )
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
            record = {
                "epoch": epoch + 1,
                "loss": total,
                "seconds": time.perf_counter() - tick,
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
    result = ad.AnnData(
        X=sp.csr_matrix(a.shape, dtype=np.float32), obs=a.obs.copy(), var=a.var.copy()
    )
    result.obsm["cell_embeddings"] = model.cell_embedding.weight.detach().cpu().numpy()
    result.varm["peak_embeddings"] = model.peak_embedding.weight.detach().cpu().numpy()
    np.save(
        out / "motif_embeddings.npy",
        model.motif_embedding.weight.detach().cpu().numpy(),
    )
    (out / "motif_names.tsv").write_text("\n".join(dataset.motif_names) + "\n")
    if calibration:
        for name in ["cell", "peak", "motif"]:
            result.uns[f"{name}_projection"] = (
                getattr(model, f"{name}_projection").weight.detach().cpu().numpy()
            )
        for name in ["cp", "pm", "cm"]:
            result.uns[f"relation_{name}"] = (
                getattr(model, f"relation_{name}").detach().cpu().numpy()
            )
        result.uns["gamma"] = model.gamma
    result.write_h5ad(out / "embedding.h5ad", compression="gzip")
    record = {
        **asdict(config),
        "stage": "calibration"
        if calibration
        else "batch_alignment"
        if batch_key
        else "embedding",
        "batch_key": batch_key,
        "seconds": time.perf_counter() - started,
    }
    (out / "config.json").write_text(json.dumps(record, indent=2) + "\n")
    return out


def train(input, bundle, output, *, epochs=20, seed=15, device=None):
    """Learn cell, peak and motif embeddings from the regulatory graph."""
    return _train(
        input, bundle, output, TrainConfig(epochs=epochs, seed=seed), device=device
    )


def calibrate(input, bundle, parent, output, *, epochs=8, seed=15, device=None):
    """Learn the relation-specific calibration on top of fixed embeddings."""
    return _train(
        input,
        bundle,
        output,
        TrainConfig(epochs=epochs, seed=seed),
        parent=parent,
        calibration=True,
        device=device,
    )


def align_batches(
    input, bundle, parent, output, *, batch_key, epochs=10, seed=15, device=None
):
    """scPRIME-BC: continue training with cross-batch anchors pulling matched cells together."""
    return _train(
        input,
        bundle,
        output,
        TrainConfig(
            epochs=epochs, seed=seed, pm_loss_weight=0.0, adaptive_negatives=False
        ),
        parent=parent,
        batch_key=batch_key,
        device=device,
    )


def _inference_input(input, bundle, model):
    a = ad.read_h5ad(input)
    m, motifs = _read_model(model)
    if not a.obs_names.equals(m.obs_names) or not a.var_names.equals(m.var_names):
        raise ValueError("Inference axes differ from model")
    names = np.asarray((Path(model) / "motif_names.tsv").read_text().splitlines())
    trained = np.asarray((Path(bundle) / "motif_names.tsv").read_text().splitlines())
    if not np.array_equal(names, trained):
        raise ValueError("Model and bundle motif axes differ")
    return a, m, motifs, names


def infer_regulators(
    input, bundle, model, family_map, output, *, label_key="celltype", exclude_labels=()
):
    """Rank regulator families for each cell type and export the peaks supporting each query."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    a, m, motifs, names = _inference_input(input, bundle, model)
    mapping = pd.read_csv(family_map, sep="\t")
    if "motif_name" not in mapping:
        raise ValueError("Family map requires motif_name and family columns")
    ids = {
        f: pd.Index(names).get_indexer(g.motif_name.dropna())
        for f, g in mapping.groupby("family", sort=False)
    }
    if any((v < 0).any() for v in ids.values()):
        raise ValueError("Family map includes motifs absent from the motif vocabulary")
    labels = a.obs[label_key].astype(str).to_numpy()
    celltypes = sorted(set(labels) - set(exclude_labels))
    matrix = sp.csr_matrix(a.X)
    stats = {}
    for ct in celltypes:
        inside = matrix[labels == ct]
        outside = matrix[labels != ct]
        if outside.shape[0] == 0:
            raise ValueError("Regulator inference requires at least two cell types")
        mean1 = np.asarray(inside.mean(axis=0)).ravel().astype(np.float64)
        mean0 = np.asarray(outside.mean(axis=0)).ravel().astype(np.float64)
        stats[ct] = {
            "specificity": (zscore(np.log1p(mean1)) - zscore(np.log1p(mean0))).astype(
                np.float32
            )
        }
    arrays = {
        "peak_embeddings": np.asarray(m.varm["peak_embeddings"], np.float32),
        "motif_embeddings": motifs,
        "peak_proj": m.varm["peak_embeddings"] @ m.uns["peak_projection"].T,
        "motif_proj": motifs @ m.uns["motif_projection"].T,
        "relation_pm": m.uns["relation_pm"],
        "gamma": float(m.uns["gamma"]),
    }
    scores = path_scores(stats, arrays, ids, celltypes)
    pm_train = sp.load_npz(Path(bundle) / "peak_motif_train.npz").tocsr()
    linked = {
        f: np.asarray(pm_train[:, ix].sum(axis=1)).ravel() > 0 for f, ix in ids.items()
    }
    # Peak-motif scores do not depend on the cell type, so they are computed once for
    # all peaks and shared by every query when listing supporting peaks.
    dot = np.empty((a.n_vars, len(names)), dtype=np.float32)
    rel = np.empty_like(dot)
    blocks = [(i, min(i + 1024, a.n_vars)) for i in range(0, a.n_vars, 1024)]

    def score_block(bounds):
        start, end = bounds
        peak_ids = np.arange(start, end, dtype=np.int64)
        return (
            start,
            end,
            rank_z_rows(embedding_scores(arrays, peak_ids)),
            rank_z_rows(relation_scores(arrays, peak_ids)),
        )

    with ThreadPoolExecutor(max_workers=min(4, len(blocks))) as pool:
        for start, end, d, r in pool.map(score_block, blocks):
            dot[start:end], rel[start:end] = d, r
    ranks = []
    support = []
    for ct in celltypes:
        ordered = sorted(
            scores[ct],
            key=lambda f: (
                not np.isfinite(scores[ct][f]),
                -scores[ct][f] if np.isfinite(scores[ct][f]) else 0.0,
                f,
            ),
        )
        specificity = stats[ct]["specificity"]
        # Supporting peaks are taken from the specific peaks that enter the first term of
        # the family score, i.e. the N_s most specific peaks of the cell type.
        specific = np.zeros(a.n_vars, dtype=bool)
        specific[np.argsort(-specificity, kind="mergesort")[:N_SPECIFIC_PEAKS]] = True
        for rank, family in enumerate(ordered, 1):
            ranks.append(
                {
                    "cell_type": ct,
                    "family": family,
                    "rank": rank,
                    "score": scores[ct][family],
                    "axis_available": bool(len(ids[family])),
                }
            )
            if not len(ids[family]):
                continue
            family_path = (dot[:, ids[family]] + rel[:, ids[family]]).max(axis=1)
            contributions = specificity + family_path
            candidates = np.flatnonzero(linked[family] & specific)
            for k, ix in enumerate(
                candidates[
                    np.argsort(-contributions[candidates], kind="mergesort")[:50]
                ],
                1,
            ):
                driver = ids[family][
                    np.argmax(dot[ix, ids[family]] + rel[ix, ids[family]])
                ]
                support.append(
                    {
                        "cell_type": ct,
                        "family": family,
                        "motif_name": names[driver],
                        "support_rank": k,
                        "peak": str(a.var_names[ix]),
                        "path_score": float(contributions[ix]),
                        "specificity": float(specificity[ix]),
                        "embedding_rank_z": float(dot[ix, driver]),
                        "relation_rank_z": float(rel[ix, driver]),
                    }
                )
    pd.DataFrame(ranks).to_csv(out / "family_rankings.tsv", sep="\t", index=False)
    pd.DataFrame(support).to_csv(out / "supporting_peaks.tsv", sep="\t", index=False)
    return out


def tf_activity(input, bundle, model, output):
    """Single-cell TF activity M(c, m): motif-count-weighted peak/motif cosine over accessible peaks."""
    from .annotations import read_scan

    if Path(output).exists():
        raise FileExistsError(output)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    a, m, motifs, names = _inference_input(input, bundle, model)
    x = sp.csr_matrix(a.X, dtype=np.float32)
    scan = read_scan(bundle, a.var_names, names)
    peaks = np.asarray(m.varm["peak_embeddings"], np.float32)
    scores = np.zeros((a.n_obs, len(names)), np.float32)
    for start in range(0, len(names), 64):
        end = min(start + 64, len(names))
        block = scan[:, start:end].tocsr()
        denom = np.clip(
            np.linalg.norm(peaks, axis=1, keepdims=True)
            * np.linalg.norm(motifs[start:end], axis=1, keepdims=True).T,
            1e-12,
            None,
        )
        cos = (peaks @ motifs[start:end].T) / denom
        weighted = block.multiply(cos).tocsr()
        numerator = np.asarray((x @ weighted).toarray(), np.float32)
        denominator = np.asarray((x @ block).toarray(), np.float32)
        np.divide(
            numerator, denominator, out=scores[:, start:end], where=denominator > 0
        )
    result = ad.AnnData(X=scores, obs=a.obs.copy(), var=pd.DataFrame(index=names))
    result.uns["score_definition"] = (
        "Accessibility- and motif-count-weighted peak/motif cosine mean; unnormalized"
    )
    result.write_h5ad(output, compression="gzip")
    return Path(output)
