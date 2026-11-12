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


# ---- paired comparison of two runs (used by `compare`) ----

def _log_binom_pmf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p))


def binom_pmf(k: int, n: int, p: float) -> float:
    return math.exp(_log_binom_pmf(k, n, p)) if 0 <= k <= n else 0.0


def _sum_down(k: int, n: int, p: float) -> float:
    """P(X <= k), summing from k downward; used only when k is at or below the mode (terms shrink)."""
    t = binom_pmf(k, n, p)
    total, i = t, k
    r = (1 - p) / p
    while i > 0 and t > total * 1e-17:
        t *= i / (n - i + 1) * r
        i -= 1
        total += t
    return total


def _sum_up(k: int, n: int, p: float) -> float:
    """P(X >= k), summing from k upward; used only when k is at or above the mode."""
    t = binom_pmf(k, n, p)
    total, i = t, k
    r = p / (1 - p)
    while i < n and t > total * 1e-17:
        t *= (n - i) / (i + 1) * r
        i += 1
        total += t
    return total


def binom_cdf(k: int, n: int, p: float) -> float:
    """Exact P(X <= k), X ~ Bin(n, p), to double precision; sums only the shrinking side of the mode."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    mode = math.floor((n + 1) * p)
    if k <= mode:
        return min(1.0, _sum_down(k, n, p))
    return max(0.0, 1.0 - _sum_up(k + 1, n, p))


def binom_sf(k: int, n: int, p: float) -> float:
    """Exact P(X >= k)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    mode = math.floor((n + 1) * p)
    if k >= mode:
        return min(1.0, _sum_up(k, n, p))
    return max(0.0, 1.0 - _sum_down(k - 1, n, p))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact (binomial) two-sided McNemar test on the discordant pairs: under 'no difference' each of the
    b + c flips is equally likely to go either way, so p = min(1, 2 * P(X <= min(b, c))), X ~ Bin(b + c, 1/2)."""
    m = b + c
    if m == 0:
        return 1.0
    return min(1.0, 2.0 * binom_cdf(min(b, c), m, 0.5))


def paired_diff_ci(n11: int, n10: int, n01: int, n00: int, z: float = Z95):
    """Tango (1998) asymptotic score interval for p_A - p_B from paired binary data
    (n10 = A right / B wrong, n01 = A wrong / B right), one of the intervals recommended by Fagerland,
    Lydersen & Laake (2014, Stat Med 33:2850). Unlike simpler intervals it does not collapse to zero width
    when there are no flips. Same formulas as R contingencytables::Tango_asymptotic_score_CI_paired_2x2."""
    n = n11 + n10 + n01 + n00
    if n == 0:
        return None, None, None
    d = (n10 - n01) / n

    def t(delta):
        A = 2 * n
        B = -n10 - n01 + (2 * n - n10 + n01) * delta
        C = -n01 * delta * (1 - delta)
        p01 = (math.sqrt(max(0.0, B * B - 4 * A * C)) - B) / (2 * A)
        v = n * (2 * p01 + delta * (1 - delta))
        num = n10 - n01 - n * delta
        return num / math.sqrt(v) if v > 0 else (0.0 if num == 0 else math.copysign(math.inf, num))

    def root(lo, hi, target):          # t is decreasing in delta
        for _ in range(200):
            mid = (lo + hi) / 2
            if t(mid) > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    eps = 1e-9
    lo = -1.0 if d <= -1 + eps or t(-1 + eps) <= z else root(-1 + eps, d, z)
    hi = 1.0 if d >= 1 - eps or t(1 - eps) >= -z else root(d, 1 - eps, -z)
    return d, lo, hi


_REJECT_CACHE: dict = {}


def _mcnemar_reject_k(m: int, alpha: float) -> int:
    """Largest k with 2 * P(X <= k) <= alpha for X ~ Bin(m, 1/2); -1 if none. The exact test rejects
    when min(b, c) <= k. Exact: a normal-approximation guess, then corrected with exact tails."""
    key = (m, alpha)
    if key in _REJECT_CACHE:
        return _REJECT_CACHE[key]
    z = _z_upper(alpha / 2)
    k = max(-1, min(m // 2, math.floor(m / 2 - z * math.sqrt(m) / 2)))
    ok = lambda j: j >= 0 and 2 * binom_cdf(j, m, 0.5) <= alpha * (1 + 1e-12)
    while k >= 0 and not ok(k):
        k -= 1
    while k + 1 <= m // 2 and ok(k + 1):
        k += 1
    _REJECT_CACHE[key] = k
    return k


def _z_upper(a: float) -> float:
    """z with P(Z > z) = a, by bisection (no scipy)."""
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if 0.5 * math.erfc(mid / math.sqrt(2)) > a:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def mcnemar_power(n: int, p10: float, p01: float, alpha: float = 0.05) -> float:
    """Exact power of the two-sided exact McNemar test with n independent pairs, where a pair is
    A-right/B-wrong with probability p10 and A-wrong/B-right with probability p01. Sums over the number of
    discordant pairs m ~ Bin(n, p10 + p01) and, given m, over b ~ Bin(m, p10 / (p10 + p01)). No normal
    approximation (terms below 1e-15 of probability are skipped)."""
    psi = p10 + p01
    if psi <= 0 or n <= 0:
        return 0.0
    q = p10 / psi
    mu, sd = n * psi, math.sqrt(n * psi * (1 - psi))
    lo, hi = max(0, int(mu - 10 * sd - 10)), min(n, int(mu + 10 * sd + 10))
    total = 0.0
    for m in range(lo, hi + 1):
        pm = binom_pmf(m, n, psi)
        if pm < 1e-15:
            continue
        k = _mcnemar_reject_k(m, alpha)
        if k < 0:
            continue
        # reject when b <= k or b >= m - k (b = A-right/B-wrong count given m discordant)
        r = binom_cdf(k, m, q) + binom_sf(m - k, m, q)
        total += pm * min(1.0, r)
    return min(1.0, total)


def n_for_drop(delta: float, psi: float, alpha: float = 0.05, power: float = 0.8, n_max: int = 1_000_000):
    """Planning estimate: the number of independent paired questions at which the exact McNemar test's
    power to detect a true accuracy difference `delta` crosses the target, if a share `psi` of questions
    flips between the runs. Found by doubling then bisection to one question (power(n - 1) < target <=
    power(n)). Exact power is saw-toothed in n, so a few nearby n may sit just above or below the target:
    read it as 'about'. Needs psi >= delta. None if above n_max."""
    if not (0 < delta <= 1) or psi < delta - 1e-12 or psi > 1 or not (0 < alpha < 1) or not (0 < power < 1):
        return None
    p10, p01 = (psi + delta) / 2, (psi - delta) / 2
    pw = lambda n: mcnemar_power(n, p10, p01, alpha)
    lo, hi = 0, 16
    while pw(hi) < power:
        if hi >= n_max:
            return None
        lo, hi = hi, min(hi * 2, n_max)
    while hi - lo > 1:                         # bisection down to a single question: pw(lo) < target <= pw(hi)
        mid = (lo + hi) // 2
        if pw(mid) >= power:
            hi = mid
        else:
            lo = mid
    return hi
