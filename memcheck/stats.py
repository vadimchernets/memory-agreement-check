"""Small, dependency-free statistics: Wilson intervals, a seeded bootstrap, the phi coefficient."""
from __future__ import annotations

import math
import random
from typing import Callable, Sequence

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95):
    if n == 0:
        return None, None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def phi(x: Sequence[bool], y: Sequence[bool]):
    """Pearson correlation of two binary vectors (here: 'model A wrong', 'model B wrong')."""
    n = len(x)
    if n == 0:
        return None
    a = sum(1 for i in range(n) if x[i] and y[i])
    b = sum(1 for i in range(n) if x[i] and not y[i])
    c = sum(1 for i in range(n) if not x[i] and y[i])
    d = n - a - b - c
    den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return None if den == 0 else (a * d - b * c) / den


def bootstrap(items: Sequence, stat: Callable[[Sequence], float | None], reps: int = 2000, seed: int = 0,
              strata: Callable | None = None):
    """Percentile 95% CI. With `strata`, resample within each stratum (keeps the design's category mix)."""
    rng = random.Random(seed)
    groups: dict = {}
    for it in items:
        groups.setdefault(strata(it) if strata else 0, []).append(it)
    vals = []
    for _ in range(reps):
        s = []
        for g in groups.values():
            s += [g[rng.randrange(len(g))] for _ in g]
        v = stat(s)
        if v is not None:
            vals.append(v)
    if len(vals) < reps * 0.5:
        return None, None
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))]
