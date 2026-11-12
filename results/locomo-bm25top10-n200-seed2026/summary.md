# Summary: locomo-bm25top10-n200-seed2026

- Model A: `cli:codex:gpt-6.1-sol`; model B: `cli:claude:sonnet`; judge: `cli:kimi:kimi-code/kimi-for-coding`
- Resolved models: {'a': ['gpt-6.1-sol'], 'b': ['claude-sonnet-5-5'], 'judge': ['kimi-code/kimi-for-coding']}; CLI versions: {'codex': 'codex-cli 0.166.0', 'claude': '2.1.322 (Claude Code)', 'kimi': 'kimi version 2.4.1'}
- Memory: `bm25-top10`; questions: 200 (seed 2026, equal number per category); LoCoMo commit `cbfbc1dba6`
- Retrieval of LoCoMo evidence turns: {'full': 95, 'partial': 27, 'none': 74, 'unknown': 4}. Questions with no usable evidence ids (left out of the retrieval split): conv-50/39, conv-50/42, conv-26/30, conv-26/46
- "Agree" = the judge (or, in the rule view, token F1 between the two answers >= 0.5) says the two answers say the same thing. "Wrong when they agree" = they agree and both are graded wrong. "Real answer" = not both "Not mentioned".
- Intervals: Wilson 95% for proportions (conditional ones treat the number of agreeing pairs as fixed); stratified bootstrap 95% (2,000 resamples) for φ and differences. Cells with n < 5 show counts only.
- φ is the correlation of the two models' error indicators. 2/(1+φ) is a heuristic for how many independent checks the pair is worth (the design-effect formula), not a measured quantity.

## View 1: third-company LLM judge

Grades from: {'judge': 200, 'rule': 0}

### Overall and by category

| group | n | model A correct | model B correct | agree | wrong when they agree on a real answer | wrong when they agree (incl. both "Not mentioned") | both wrong (observed vs if independent) | error correlation φ |
|---|---|---|---|---|---|---|---|---|
| all questions | 200 | 59.5% [52.6, 66.1] | 64.0% [57.1, 70.3] | 84.5% [78.8, 88.9] | 19.2% [12.6, 28.0] (19/99) | 33.1% [26.5, 40.5] (56/169) | 33.0% [26.9, 39.8] vs 14.6% | 0.78 [0.69, 0.87] |
| all except adversarial | 160 | 51.2% [43.6, 58.9] | 58.8% [51.0, 66.1] | 83.8% [77.3, 88.7] | 17.5% [11.2, 26.3] (17/97) | 40.3% [32.4, 48.8] (54/134) | 40.0% [32.7, 47.7] vs 20.1% | 0.81 [0.72, 0.88] |
| multi-hop | 40 | 35.0% [22.1, 50.5] | 32.5% [20.1, 48.0] | 87.5% [73.9, 94.5] | 40.9% [23.3, 61.3] (9/22) | 62.9% [46.3, 76.8] (22/35) | 65.0% [49.5, 77.9] vs 43.9% | 0.95 [0.83, 1.00] |
| temporal | 40 | 70.0% [54.6, 81.9] | 72.5% [57.2, 83.9] | 90.0% [76.9, 96.0] | 9.7% [3.3, 24.9] (3/31) | 22.2% [11.7, 38.1] (8/36) | 27.5% [16.1, 42.8] vs 8.3% | 0.94 [0.80, 1.00] |
| open-domain | 40 | 30.0% [18.1, 45.4] | 52.5% [37.5, 67.1] | 72.5% [57.2, 83.9] | 14.3% [4.0, 39.9] (2/14) | 58.6% [40.7, 74.5] (17/29) | 47.5% [32.9, 62.5] vs 33.2% | 0.62 [0.45, 0.81] |
| single-hop | 40 | 70.0% [54.6, 81.9] | 77.5% [62.5, 87.7] | 85.0% [70.9, 92.9] | 10.0% [3.5, 25.6] (3/30) | 20.6% [10.3, 36.8] (7/34) | 20.0% [10.5, 34.8] vs 6.8% | 0.69 [0.42, 0.90] |
| adversarial | 40 | 92.5% [80.1, 97.4] | 85.0% [70.9, 92.9] | 87.5% [73.9, 94.5] | 2/2 | 5.7% [1.6, 18.6] (2/35) | 5.0% [1.4, 16.5] vs 1.1% | 0.41 [-0.08, 0.81] |

### By retrieval quality (categories 1-4)

| group | n | model A correct | model B correct | agree | wrong when they agree on a real answer | wrong when they agree (incl. both "Not mentioned") | both wrong (observed vs if independent) | error correlation φ |
|---|---|---|---|---|---|---|---|---|
| evidence fully retrieved | 71 | 88.7% [79.3, 94.2] | 94.4% [86.4, 97.8] | 90.1% [81.0, 95.1] | 3.2% [0.9, 10.9] (2/63) | 4.7% [1.6, 12.9] (3/64) | 4.2% [1.4, 11.7] vs 0.6% | 0.49 [-0.03, 0.80] |
| evidence incomplete | 85 | 21.2% [13.8, 31.0] | 28.2% [19.8, 38.6] | 80.0% [70.3, 87.1] | 45.5% [29.8, 62.0] (15/33) | 73.5% [62.0, 82.6] (50/68) | 70.6% [60.2, 79.2] vs 56.6% | 0.76 [0.60, 0.90] |

Within each category, agreed-and-wrong / agreed:

| category | full memory: real answer | full: incl. "Not mentioned" | incomplete: real answer | incomplete: incl. "Not mentioned" |
|---|---|---|---|---|
| multi-hop | 0/6 (0%) | 0/6 (0%) | 9/16 (56%) | 22/29 (76%) |
| temporal | 1/26 (4%) | 1/26 (4%) | 2/5 (40%) | 7/10 (70%) |
| open-domain | 0/4 | 1/5 (20%) | 2/9 (22%) | 15/22 (68%) |
| single-hop | 1/27 (4%) | 1/27 (4%) | 2/3 | 6/7 (86%) |

- Wrong-when-agreeing on a real answer, incomplete minus full retrieval: pooled +42.3 points [+24.3, +59.3]; category-standardized +44.4 points [+20.7, +64.6]
- Same including both-"Not mentioned" pairs: pooled +68.8 points [+56.0, +79.7]; category-standardized +68.1 points [+51.6, +81.8]
- Agreed-but-wrong pairs (categories 1-4): 54, of which both "Not mentioned": 37
- Share of agreed-but-wrong pairs with incomplete retrieval: 92.6% [82.4, 97.1]; real answers only: 88.2% [65.7, 96.7]

## View 2: LoCoMo token-F1 rule (correct = F1 >= 0.5; adversarial = says "not mentioned")

Grades from: {'judge': 0, 'rule': 200}

### Overall and by category

| group | n | model A correct | model B correct | agree | wrong when they agree on a real answer | wrong when they agree (incl. both "Not mentioned") | both wrong (observed vs if independent) | error correlation φ |
|---|---|---|---|---|---|---|---|---|
| all questions | 200 | 53.5% [46.6, 60.3] | 51.5% [44.6, 58.3] | 79.0% [72.8, 84.1] | 31.8% [23.0, 42.1] (28/88) | 41.1% [33.8, 48.9] (65/158) | 41.0% [34.4, 47.9] vs 22.6% | 0.74 [0.65, 0.82] |
| all except adversarial | 160 | 43.8% [36.3, 51.5] | 43.1% [35.7, 50.9] | 76.9% [69.8, 82.7] | 30.2% [21.5, 40.6] (26/86) | 51.2% [42.5, 59.9] (63/123) | 50.0% [42.3, 57.7] vs 32.0% | 0.73 [0.62, 0.83] |
| multi-hop | 40 | 22.5% [12.3, 37.5] | 17.5% [8.7, 31.9] | 82.5% [68.1, 91.3] | 65.0% [43.3, 81.9] (13/20) | 78.8% [62.2, 89.3] (26/33) | 77.5% [62.5, 87.7] vs 63.9% | 0.85 [0.65, 1.00] |
| temporal | 40 | 67.5% [52.0, 79.9] | 65.0% [49.5, 77.9] | 87.5% [73.9, 94.5] | 13.3% [5.3, 29.7] (4/30) | 25.7% [14.2, 42.1] (9/35) | 32.5% [20.1, 48.0] vs 11.4% | 0.95 [0.83, 1.00] |
| open-domain | 40 | 22.5% [12.3, 37.5] | 32.5% [20.1, 48.0] | 65.0% [49.5, 77.9] | 45.5% [21.3, 72.0] (5/11) | 76.9% [57.9, 89.0] (20/26) | 60.0% [44.6, 73.7] vs 52.3% | 0.39 [0.07, 0.69] |
| single-hop | 40 | 62.5% [47.0, 75.8] | 57.5% [42.2, 71.5] | 72.5% [57.2, 83.9] | 16.0% [6.4, 34.7] (4/25) | 27.6% [14.7, 45.7] (8/29) | 30.0% [18.1, 45.4] vs 15.9% | 0.59 [0.31, 0.83] |
| adversarial | 40 | 92.5% [80.1, 97.4] | 85.0% [70.9, 92.9] | 87.5% [73.9, 94.5] | 2/2 | 5.7% [1.6, 18.6] (2/35) | 5.0% [1.4, 16.5] vs 1.1% | 0.41 [-0.08, 0.81] |

### By retrieval quality (categories 1-4)

| group | n | model A correct | model B correct | agree | wrong when they agree on a real answer | wrong when they agree (incl. both "Not mentioned") | both wrong (observed vs if independent) | error correlation φ |
|---|---|---|---|---|---|---|---|---|
| evidence fully retrieved | 71 | 77.5% [66.5, 85.6] | 74.6% [63.4, 83.3] | 81.7% [71.2, 89.0] | 12.3% [6.1, 23.2] (7/57) | 13.8% [7.2, 24.9] (8/58) | 16.9% [9.9, 27.3] vs 5.7% | 0.62 [0.38, 0.82] |
| evidence incomplete | 85 | 16.5% [10.1, 25.8] | 16.5% [10.1, 25.8] | 75.3% [65.2, 83.2] | 65.5% [47.3, 80.1] (19/29) | 84.4% [73.6, 91.3] (54/64) | 78.8% [69.0, 86.2] vs 69.8% | 0.66 [0.42, 0.85] |

Within each category, agreed-and-wrong / agreed:

| category | full memory: real answer | full: incl. "Not mentioned" | incomplete: real answer | incomplete: incl. "Not mentioned" |
|---|---|---|---|---|
| multi-hop | 2/6 (33%) | 2/6 (33%) | 11/14 (79%) | 24/27 (89%) |
| temporal | 2/25 (8%) | 2/25 (8%) | 2/5 (40%) | 7/10 (70%) |
| open-domain | 0/3 | 1/4 | 5/8 (62%) | 18/21 (86%) |
| single-hop | 3/23 (13%) | 3/23 (13%) | 1/2 | 5/6 (83%) |

- Wrong-when-agreeing on a real answer, incomplete minus full retrieval: pooled +53.2 points [+33.8, +71.1]; category-standardized +44.2 points [+18.5, +69.4]
- Same including both-"Not mentioned" pairs: pooled +70.6 points [+57.6, +81.9]; category-standardized +62.1 points [+42.7, +79.8]
- Agreed-but-wrong pairs (categories 1-4): 63, of which both "Not mentioned": 37
- Share of agreed-but-wrong pairs with incomplete retrieval: 85.7% [75.0, 92.3]; real answers only: 73.1% [53.9, 86.3]

Mean LoCoMo F1 (the official metric) over all questions: model A 0.539, model B 0.537.
