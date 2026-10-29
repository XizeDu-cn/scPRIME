"""Independent random streams so that each sampling step is reproducible on its own."""

import random
import numpy as np
import torch


class Streams:
    def __init__(self, seed, device):
        self.perm = np.random.default_rng(seed + 101)
        self.anchor = np.random.default_rng(seed + 202)
        self.order = np.random.default_rng(seed + 303)
        self.python = random.Random(seed + 404)
        self.pm = np.random.default_rng(seed + 707)
        self.cpu = torch.Generator(device="cpu").manual_seed(seed + 505)
        self.gpu = (
            torch.Generator(device=device).manual_seed(seed + 505)
            if device.type == "cuda"
            else None
        )

    def generator_for_device(self, device):
        return self.gpu if torch.device(device).type == "cuda" else self.cpu


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
