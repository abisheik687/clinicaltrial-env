"""Shared seed management helpers."""

import random

import numpy as np


class SeedManager:
    """Centralized deterministic seeding."""

    @staticmethod
    def seed_all(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)

