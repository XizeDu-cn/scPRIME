"""Embedding objectives of scPRIME."""

from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

# Score assigned to negatives that must never win the softmax (observed edges, padding).
MASKED_SCORE = -1000000000.0


class EmbeddingModel(nn.Module):
    def __init__(self, dataset, config):
        super().__init__()
        self.dataset = dataset
        self.num_cells = dataset.n_cells
        self.num_peaks = dataset.m_peaks
        self.num_motifs = dataset.n_motifs
        self.embedding_dim = config.embedding_dim
        self.adaptive_negatives = config.adaptive_negatives
        self.adv_pool_size = config.adv_pool_size
        self.adv_alpha = config.adv_alpha
        self.cell_embedding = nn.Embedding(
            self.num_cells, self.embedding_dim, sparse=True
        )
        self.peak_embedding = nn.Embedding(
            self.num_peaks, self.embedding_dim, sparse=True
        )
        self.motif_embedding = nn.Embedding(
            self.num_motifs, self.embedding_dim, sparse=True
        )
        self.rng_streams = None
        for layer in [self.cell_embedding, self.peak_embedding, self.motif_embedding]:
            nn.init.uniform_(
                layer.weight, -0.5 / self.embedding_dim, 0.5 / self.embedding_dim
            )

    # ------------------------------------------------------------------ objectives

    def alignment_loss(
        self,
        anchor_i: torch.Tensor,
        anchor_j: torch.Tensor,
        anchor_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if anchor_i.numel() == 0:
            return self.cell_embedding.weight.new_zeros(())
        z_i = self.cell_embedding(anchor_i)
        z_j = self.cell_embedding(anchor_j)
        sqdist = (z_i - z_j).pow(2).sum(dim=1)
        if anchor_w is not None:
            return (anchor_w * sqdist).mean()
        return sqdist.mean()

    def cell_peak_loss(
        self,
        cells: torch.Tensor,
        peaks: torch.Tensor,
        num_batch_negs: int,
        num_uniform_negs: int,
    ) -> torch.Tensor:
        """Bidirectional softmax loss over cell-peak edges.

        Each edge is contrasted with negative peaks for the cell and negative cells
        for the peak. Negatives come from the other edges of the same chunk (batch
        negatives) plus sampled entities; sampled negative peaks are drawn adaptively
        when ``adaptive_negatives`` is set.
        """
        num_pos = len(cells)
        if num_pos == 0:
            return self.cell_embedding.weight.new_zeros(())
        e_cells = self.cell_embedding(cells)
        e_peaks = self.peak_embedding(peaks)
        chunk_size, use_batch_negs = self._resolve_chunk_size(
            num_pos, num_batch_negs, num_uniform_negs
        )
        e_cells_chunk = self._chunk_embeddings(e_cells, chunk_size)
        e_peaks_chunk = self._chunk_embeddings(e_peaks, chunk_size)
        cells_chunk = self._chunk_entity_ids(cells, chunk_size)
        peaks_chunk = self._chunk_entity_ids(peaks, chunk_size)
        pos_scores = (e_cells_chunk * e_peaks_chunk).sum(-1)
        peak_batch_scores = e_cells_chunk.new_empty(
            (e_cells_chunk.size(0), e_cells_chunk.size(1), 0)
        )
        if use_batch_negs:
            peak_batch_scores = torch.bmm(e_cells_chunk, e_peaks_chunk.transpose(1, 2))
            self._mask_chunked_batch_negatives(peak_batch_scores, num_pos, chunk_size)
            self._mask_peak_batch_accidental_hits(
                peak_batch_scores, cells_chunk, peaks_chunk, num_pos, chunk_size
            )
        peak_sampled_scores = self._sampled_negative_peak_scores(
            e_cells_chunk, cells_chunk, num_pos, chunk_size, num_uniform_negs
        )
        peak_neg_scores = torch.cat([peak_batch_scores, peak_sampled_scores], dim=2)
        cell_neg_embs = self._chunked_negative_embeddings(
            pos_embs=e_cells_chunk,
            num_entities=self.num_cells,
            embedding_layer=self.cell_embedding,
            num_uniform_negs=num_uniform_negs,
            use_batch_negs=use_batch_negs,
        )
        cell_neg_scores = torch.bmm(e_peaks_chunk, cell_neg_embs.transpose(1, 2))
        if use_batch_negs:
            self._mask_chunked_batch_negatives(cell_neg_scores, num_pos, chunk_size)
        pos_scores = pos_scores.flatten()[:num_pos]
        peak_neg_scores = peak_neg_scores.flatten(0, 1)[:num_pos]
        cell_neg_scores = cell_neg_scores.flatten(0, 1)[:num_pos]
        return self._softmax_loss(pos_scores, peak_neg_scores) + self._softmax_loss(
            pos_scores, cell_neg_scores
        )

    def peak_motif_loss(self, peaks: torch.Tensor, motifs: torch.Tensor) -> torch.Tensor:
        """Softmax loss of each peak-motif edge against all motifs not annotated to the peak."""
        num_pos = int(len(peaks))
        if num_pos == 0:
            return self.peak_embedding.weight.new_zeros(())
        e_peaks = self.peak_embedding(peaks)
        e_motifs = self.motif_embedding(motifs)
        pos_scores = (e_peaks * e_motifs).sum(dim=1)
        motif_ids = torch.arange(self.num_motifs, dtype=torch.long, device=peaks.device)
        neg_scores = e_peaks @ self.motif_embedding.weight.transpose(0, 1)
        annotated = self.dataset.lookup_peak_motif_membership(
            peaks.unsqueeze(1), motif_ids.unsqueeze(0)
        ).clone()
        annotated.scatter_(1, motifs.view(-1, 1), False)
        neg_scores = neg_scores.masked_fill(annotated, MASKED_SCORE)
        neg_scores.scatter_(1, motifs.view(-1, 1), MASKED_SCORE)
        return self._softmax_loss(pos_scores, neg_scores)

    def l2_norm(self) -> torch.Tensor:
        total = self.cell_embedding.weight.norm(2).pow(2) + self.peak_embedding.weight.norm(
            2
        ).pow(2)
        return total + self.motif_embedding.weight.norm(2).pow(2)

    # ------------------------------------------------------------------ negative sampling

    def _sampled_negative_peak_scores(
        self,
        e_cells_chunk: torch.Tensor,
        cells_chunk: torch.Tensor,
        num_pos: int,
        chunk_size: int,
        num_uniform_negs: int,
    ) -> torch.Tensor:
        """Scores of sampled negative peaks for each cell, shape (chunks, chunk_size, n)."""
        num_chunks = e_cells_chunk.size(0)
        if not self.adaptive_negatives:
            uniform_ids = self._torch_randint(
                0, self.num_peaks, (num_chunks, num_uniform_negs), device=e_cells_chunk.device
            )
            uniform_scores = torch.bmm(
                e_cells_chunk, self.peak_embedding(uniform_ids).transpose(1, 2)
            )
            accidental_hits = self._peak_accidental_hits(
                cells_chunk, uniform_ids, num_pos, chunk_size
            )
            uniform_scores.masked_fill_(accidental_hits, MASKED_SCORE)
            return uniform_scores
        # Adaptive sampling: draw a uniform candidate pool, then resample n negatives
        # with probability proportional to exp(alpha * score), so that peaks the model
        # currently confuses with accessible ones are chosen more often.
        candidate_pool_size = max(int(self.adv_pool_size), int(num_uniform_negs))
        candidate_ids = self._torch_randint(
            0, self.num_peaks, (num_chunks, candidate_pool_size), device=e_cells_chunk.device
        )
        candidate_scores = torch.bmm(
            e_cells_chunk, self.peak_embedding(candidate_ids).transpose(1, 2)
        )
        raw_candidate_scores = candidate_scores.clone()
        accidental_hits = self._peak_accidental_hits(
            cells_chunk, candidate_ids, num_pos, chunk_size
        )
        candidate_scores.masked_fill_(accidental_hits, MASKED_SCORE)
        # A cell whose whole pool consists of its own accessible peaks keeps the raw scores,
        # otherwise the softmax below would be undefined.
        empty_rows = (~accidental_hits).sum(dim=2) == 0
        if empty_rows.any():
            candidate_scores[empty_rows] = raw_candidate_scores[empty_rows]
        flat_scores = candidate_scores.flatten(0, 1)[:num_pos]
        sampled_indices = self._torch_multinomial(
            torch.softmax(self.adv_alpha * flat_scores, dim=1),
            num_samples=num_uniform_negs,
            replacement=True,
        )
        sampled_scores = flat_scores.gather(1, sampled_indices)
        padded_rows = num_chunks * chunk_size
        if padded_rows > num_pos:
            padding = sampled_scores.new_zeros((padded_rows - num_pos, num_uniform_negs))
            sampled_scores = torch.cat([sampled_scores, padding], dim=0)
        return sampled_scores.view(num_chunks, chunk_size, num_uniform_negs)

    def _chunked_negative_embeddings(
        self,
        pos_embs: torch.Tensor,
        num_entities: int,
        embedding_layer: nn.Embedding,
        num_uniform_negs: int,
        use_batch_negs: bool,
    ) -> torch.Tensor:
        # Positives of the same chunk double as batch negatives for each other.
        batch_neg_embs = (
            pos_embs
            if use_batch_negs
            else pos_embs.new_empty((pos_embs.size(0), 0, pos_embs.size(-1)))
        )
        neg_embs = batch_neg_embs
        if num_uniform_negs > 0:
            uniform_ids = self._torch_randint(
                0, num_entities, (pos_embs.size(0), num_uniform_negs), device=pos_embs.device
            )
            neg_embs = torch.cat([neg_embs, embedding_layer(uniform_ids)], dim=1)
        return neg_embs

    def _torch_randint(
        self, low: int, high: int, size: tuple[int, ...], *, device: torch.device | str
    ) -> torch.Tensor:
        generator = self.rng_streams.generator_for_device(device)
        return torch.randint(low, high, size, device=device, generator=generator)

    def _torch_multinomial(
        self, probs: torch.Tensor, *, num_samples: int, replacement: bool
    ) -> torch.Tensor:
        return torch.multinomial(
            probs,
            num_samples=num_samples,
            replacement=replacement,
            generator=self.rng_streams.generator_for_device(probs.device),
        )

    # ------------------------------------------------------------------ chunking and masks

    def _resolve_chunk_size(
        self, num_pos: int, num_batch_negs: int, num_uniform_negs: int
    ) -> tuple[int, bool]:
        if num_batch_negs > 0:
            return (min(num_batch_negs, num_pos), True)
        if num_uniform_negs > 0:
            return (min(num_uniform_negs, num_pos), False)
        return (num_pos, False)

    def _chunk_embeddings(self, embs: torch.Tensor, chunk_size: int) -> torch.Tensor:
        num_pos, dim = embs.shape
        num_chunks = (num_pos + chunk_size - 1) // chunk_size
        padded_size = num_chunks * chunk_size
        if padded_size > num_pos:
            padding = embs.new_zeros((padded_size - num_pos, dim))
            embs = torch.cat((embs, padding), dim=0)
        return embs.view(num_chunks, chunk_size, dim)

    def _chunk_entity_ids(self, entity_ids: torch.Tensor, chunk_size: int) -> torch.Tensor:
        num_pos = entity_ids.shape[0]
        num_chunks = (num_pos + chunk_size - 1) // chunk_size
        padded_size = num_chunks * chunk_size
        if padded_size > num_pos:
            padding = entity_ids.new_zeros(padded_size - num_pos)
            entity_ids = torch.cat((entity_ids, padding), dim=0)
        return entity_ids.view(num_chunks, chunk_size)

    def _mask_chunked_batch_negatives(
        self, neg_scores: torch.Tensor, num_pos: int, chunk_size: int
    ) -> None:
        # The diagonal is the positive itself; padded rows of the last chunk are not real edges.
        num_chunks = neg_scores.size(0)
        n_real_in_last_chunk = num_pos - (num_chunks - 1) * chunk_size
        chunk_indices = torch.arange(chunk_size, device=neg_scores.device)
        if num_chunks > 1:
            neg_scores[:-1, chunk_indices, chunk_indices] = MASKED_SCORE
        real_last_indices = chunk_indices[:n_real_in_last_chunk]
        neg_scores[-1, real_last_indices, real_last_indices] = MASKED_SCORE
        if n_real_in_last_chunk < chunk_size:
            neg_scores[-1, :n_real_in_last_chunk, n_real_in_last_chunk:chunk_size] = (
                MASKED_SCORE
            )

    def _mask_peak_batch_accidental_hits(
        self,
        peak_neg_scores: torch.Tensor,
        cells_chunk: torch.Tensor,
        peaks_chunk: torch.Tensor,
        num_pos: int,
        chunk_size: int,
    ) -> None:
        # A batch negative that is itself accessible in the cell is an observed edge, not a negative.
        accidental_hits = self._peak_accidental_hits(cells_chunk, peaks_chunk, num_pos, chunk_size)
        chunk_indices = torch.arange(chunk_size, device=peak_neg_scores.device)
        accidental_hits[:, chunk_indices, chunk_indices] = False
        n_real_in_last_chunk = num_pos - (cells_chunk.size(0) - 1) * chunk_size
        if n_real_in_last_chunk < chunk_size:
            accidental_hits[-1, :, n_real_in_last_chunk:] = False
        peak_neg_scores[:, :, :chunk_size].masked_fill_(accidental_hits, MASKED_SCORE)

    def _peak_accidental_hits(
        self,
        cells_chunk: torch.Tensor,
        candidate_peak_ids: torch.Tensor,
        num_pos: int,
        chunk_size: int,
    ) -> torch.Tensor:
        accidental_hits = self.dataset.lookup_edge_membership(
            cells_chunk.unsqueeze(-1), candidate_peak_ids.unsqueeze(1)
        )
        num_chunks = accidental_hits.size(0)
        n_real_in_last_chunk = num_pos - (num_chunks - 1) * chunk_size
        if n_real_in_last_chunk < chunk_size:
            accidental_hits = accidental_hits.clone()
            accidental_hits[-1, n_real_in_last_chunk:, :] = False
        return accidental_hits

    def _softmax_loss(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        batch_size = pos_scores.shape[0]
        if batch_size == 0 or neg_scores.shape[1] == 0:
            return torch.zeros((), device=pos_scores.device, requires_grad=True)
        neg_mass = neg_scores.logsumexp(dim=1, keepdim=True)
        scores = torch.cat([pos_scores.unsqueeze(1), neg_mass], dim=1)
        target = torch.zeros(batch_size, dtype=torch.long, device=pos_scores.device)
        return F.cross_entropy(scores, target, reduction="none").sum()
