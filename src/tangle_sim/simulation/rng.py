from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


STREAMS = ("workload", "topology", "delay", "selection", "attack", "measurement", "bootstrap")


def _word(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "little")


@dataclass(frozen=True)
class SeedManager:
    root_seed: int

    def seed_sequence(self, scenario: str, replicate: int, stream: str, algorithm: str = "common") -> np.random.SeedSequence:
        if stream not in STREAMS:
            raise ValueError(f"unknown RNG stream: {stream}")
        if stream in {"workload", "topology", "delay", "attack"}:
            algorithm = "common"
        return np.random.SeedSequence(
            entropy=self.root_seed,
            spawn_key=(_word(scenario), int(replicate), _word(stream), _word(algorithm)),
        )

    def generator(self, scenario: str, replicate: int, stream: str, algorithm: str = "common") -> np.random.Generator:
        return np.random.default_rng(self.seed_sequence(scenario, replicate, stream, algorithm))

    def identifier(self, scenario: str, replicate: int, stream: str, algorithm: str = "common") -> str:
        ss = self.seed_sequence(scenario, replicate, stream, algorithm)
        return "-".join(str(int(x)) for x in ss.generate_state(4))


def independent_generators(rng: np.random.Generator, count: int) -> list[np.random.Generator]:
    """Derive independent substreams without depending on their later consumption."""
    entropy = [int(x) for x in rng.integers(0, 2**32, size=4, dtype=np.uint32)]
    return [np.random.default_rng(child) for child in np.random.SeedSequence(entropy).spawn(count)]


@dataclass(frozen=True, slots=True)
class LazyIndependentGenerators:
    """Create deterministic child streams only when a retry needs one.

    A child built with ``spawn_key=(i,)`` has the same state as child ``i``
    returned by ``SeedSequence(entropy).spawn(n)``.  This retains the exact
    former streams without constructing all retry generators eagerly.
    """

    entropy: tuple[int, int, int, int]

    @classmethod
    def from_generator(cls, rng: np.random.Generator) -> "LazyIndependentGenerators":
        words = rng.integers(0, 2**32, size=4, dtype=np.uint32)
        return cls(tuple(int(value) for value in words))

    def __getitem__(self, index: int) -> np.random.Generator:
        if index < 0:
            raise IndexError("child stream index must be nonnegative")
        child = np.random.SeedSequence(self.entropy, spawn_key=(int(index),))
        return np.random.default_rng(child)
