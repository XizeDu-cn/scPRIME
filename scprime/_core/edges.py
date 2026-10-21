"""Sparse exact positive-edge membership lookup."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import torch


@dataclass(slots=True)
class SortedEdgeIndex:
    num_items: int
    codes_cpu: torch.Tensor
    _codes_by_device: dict[str, torch.Tensor] = field(
        default_factory=dict, init=False, repr=False
    )

    @classmethod
    def from_pairs(
        cls, cell_ids: np.ndarray, peak_ids: np.ndarray, num_items: int
    ) -> "SortedEdgeIndex":
        pair_codes = np.asarray(cell_ids, dtype=np.int64) * int(num_items) + np.asarray(
            peak_ids, dtype=np.int64
        )
        sorted_codes = torch.from_numpy(np.sort(pair_codes, kind="mergesort"))
        return cls(num_items=int(num_items), codes_cpu=sorted_codes)

    def _codes_for_device(self, device: torch.device | str) -> torch.Tensor:
        device = torch.device(device)
        if device.type == "cpu":
            return self.codes_cpu
        key = str(device)
        cached = self._codes_by_device.get(key)
        if cached is None:
            cached = self.codes_cpu.to(
                device=device, dtype=torch.int64, non_blocking=False
            )
            self._codes_by_device[key] = cached
        return cached

    def contains(self, cell_ids: torch.Tensor, peak_ids: torch.Tensor) -> torch.Tensor:
        if cell_ids.device != peak_ids.device:
            raise ValueError("cell_ids and peak_ids must be on the same device")
        codes = self._codes_for_device(cell_ids.device)
        if codes.numel() == 0:
            shape = torch.broadcast_shapes(tuple(cell_ids.shape), tuple(peak_ids.shape))
            return torch.zeros(shape, dtype=torch.bool, device=cell_ids.device)
        lhs, rhs = torch.broadcast_tensors(
            cell_ids.to(dtype=torch.int64), peak_ids.to(dtype=torch.int64)
        )
        flat_codes = lhs.reshape(-1) * self.num_items + rhs.reshape(-1)
        positions = torch.searchsorted(codes, flat_codes)
        valid_positions = positions < codes.numel()
        safe_positions = positions.clamp(max=codes.numel() - 1)
        membership = valid_positions & (codes[safe_positions] == flat_codes)
        return membership.reshape(lhs.shape)
