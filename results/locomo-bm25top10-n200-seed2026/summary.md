# Summary: locomo-bm25top10-n200-seed2026

- Model A: `cli:codex:gpt-6.1-sol`; model B: `cli:claude:sonnet`; judge: `cli:kimi:kimi-code/kimi-for-coding`
- Memory: `bm25-top10`; questions: 200 (seed 2026, equal number per category); LoCoMo commit `cbfbc1dba6`
- Correctness from: {'judge': 200, 'rule': 0}; stemmer: nltk-porter
- Retrieval of LoCoMo evidence turns: {'full': 95, 'partial': 27, 'none': 74, 'unknown': 4}
- Intervals: Wilson 95% for proportions; stratified bootstrap 95% (2,000 resamples) for φ and differences.
- "Agree" = the two answers say the same thing and get the same grade. "Real answer" = not both "Not mentioned".

## Overall and by category

| group | n | model A correct | model B correct | agree | wrong when they agree | wrong when they agree on a real answer | both wrong (observed vs if independent) | error correlation φ |
|---|---|---|---|---|---|---|---|---|
| all questions | 200 | 59.5% [52.6, 66.1] | 64.0% [57.1, 70.3] | 84.0% [78.3, 88.4] | 33.3% [26.6, 40.8] (56/168) | 19.4% [12.8, 28.3] (19/98) | 33.0% [26.9, 39.8] vs 14.6% | 0.78 [0.69, 0.87] |
| all except adversarial | 160 | 51.2% [43.6, 58.9] | 58.8% [51.0, 66.1] | 83.1% [76.6, 88.1] | 40.6% [32.6, 49.1] (54/133) | 17.7% [11.4, 26.5] (17/96) | 40.0% [32.7, 47.7] vs 20.1% | 0.81 [0.72, 0.88] |
| multi-hop | 40 | 35.0% [22.1, 50.5] | 32.5% [20.1, 48.0] | 87.5% [73.9, 94.5] | 62.9% [46.3, 76.8] (22/35) | 40.9% [23.3, 61.3] (9/22) | 65.0% [49.5, 77.9] vs 43.9% | 0.95 [0.83, 1.00] |
| temporal | 40 | 70.0% [54.6, 81.9] | 72.5% [57.2, 83.9] | 90.0% [76.9, 96.0] | 22.2% [11.7, 38.1] (8/36) | 9.7% [3.3, 24.9] (3/31) | 27.5% [16.1, 42.8] vs 8.3% | 0.94 [0.80, 1.00] |
| open-domain | 40 | 30.0% [18.1, 45.4] | 52.5% [37.5, 67.1] | 72.5% [57.2, 83.9] | 58.6% [40.7, 74.5] (17/29) | 14.3% [4.0, 39.9] (2/14) | 47.5% [32.9, 62.5] vs 33.2% | 0.62 [0.45, 0.81] |
| single-hop | 40 | 70.0% [54.6, 81.9] | 77.5% [62.5, 87.7] | 82.5% [68.1, 91.3] | 21.2% [10.7, 37.8] (7/33) | 10.3% [3.6, 26.4] (3/29) | 20.0% [10.5, 34.8] vs 6.8% | 0.69 [0.42, 0.90] |
| adversarial | 40 | 92.5% [80.1, 97.4] | 85.0% [70.9, 92.9] | 87.5% [73.9, 94.5] | 5.7% [1.6, 18.6] (2/35) | 100.0% [34.2, 100.0] (2/2) | 5.0% [1.4, 16.5] vs 1.1% | 0.41 [-0.08, 0.81] |

## By retrieval quality (categories 1-4)

"Full": every evidence turn LoCoMo marks for the question was in the retrieved memory. "Incomplete": at least one was missing.

| group | n | model A correct | model B correct | agree | wrong when they agree | wrong when they agree on a real answer | both wrong (observed vs if independent) | error correlation φ |
|---|---|---|---|---|---|---|---|---|
| evidence fully retrieved | 71 | 88.7% [79.3, 94.2] | 94.4% [86.4, 97.8] | 90.1% [81.0, 95.1] | 4.7% [1.6, 12.9] (3/64) | 3.2% [0.9, 10.9] (2/63) | 4.2% [1.4, 11.7] vs 0.6% | 0.49 [-0.03, 0.80] |
| evidence incomplete | 85 | 21.2% [13.8, 31.0] | 28.2% [19.8, 38.6] | 78.8% [69.0, 86.2] | 74.6% [63.1, 83.5] (50/67) | 46.9% [30.9, 63.6] (15/32) | 70.6% [60.2, 79.2] vs 56.6% | 0.76 [0.60, 0.90] |

- Wrong-when-agreeing, incomplete minus full retrieval: +69.9 points [+58.2, +80.7]
- Same, counting only agreement on a real answer: +43.7 points [+25.2, +61.5]
- Share of agreed-but-wrong answers (categories 1-4) that came with incomplete retrieval: 92.6% [82.4, 97.1]
- Same, agreement on a real answer only: 88.2% [65.7, 96.7]
