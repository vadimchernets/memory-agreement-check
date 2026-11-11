# memory-agreement-check

**When two AI models from different companies agree, that is often taken as a sign the answer is right. In an agent with memory, both models usually read the same retrieved memory, so if retrieval missed the fact they can agree on the same wrong answer.**

This tool measures how often that happens, on the [LoCoMo](https://github.com/snap-research/locomo) long-conversation benchmark. For each question it:

1. retrieves memory: a built-in BM25 "simple memory" over the dialogue, or contexts from **your own retriever** (JSONL);
2. asks **two models from different companies separately** (same prompt, neither sees the other's answer);
3. grades both answers: LoCoMo's official token-F1 rules, plus an **LLM judge from a third company** when available;
4. reports each model's accuracy, how often they agree, how often they **agree and are both wrong**, by LoCoMo category, and separately for questions where **retrieval missed the evidence** that LoCoMo marks for the question.

Results of the first run are under [`results/`](results/). Headline numbers are [below](#first-numbers).

## Install

Python 3.10+, no required dependencies. Optional: `pip install nltk` (the official LoCoMo scorer stems words with NLTK's Porter stemmer; without it F1 is computed without stemming and the run records `stemmer: none`).

```bash
git clone https://github.com/vadimchernets/memory-agreement-check
cd memory-agreement-check
python -m memcheck download          # fetches LoCoMo at a pinned commit and checks its SHA-256
python -m unittest discover -s tests -t .
```

## Run it: three backends

**1. Your subscriptions (no API keys).** Uses the `codex` CLI (ChatGPT plan) and the `claude` CLI (Claude plan), with the `kimi` CLI as the third-company judge. Codex runs with a read-only sandbox; Claude runs with all tools disabled. Each answering prompt tells the model to use only the excerpts.

```bash
python -m memcheck run --backend cli --n 200 --seed 2026 \
  --model-a cli:codex:gpt-6.1-sol --model-b cli:claude:sonnet --judge cli:kimi
```

**2. APIs (no subscriptions).** Keys from `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`; name the models yourself:

```bash
python -m memcheck run --backend api \
  --model-a api:openai:<openai-model> --model-b api:anthropic:<claude-model> \
  --judge rule          # or a third model, e.g. --judge api:openai:<another-model>
```

**3. Dry (offline).** Deterministic stub models, to try the pipeline in seconds. The default stubs answer correctly only when memory holds the evidence and otherwise give the same plausible wrong answer, which is the failure this tool looks for:

```bash
python -m memcheck run --backend dry --n 200
```

Useful flags: `--k` (turns retrieved by BM25, default 10), `--categories 1,2,3,4`, `--random` (simple random sample instead of equal per category), `--workers` and `--pace` (be gentle with subscription limits), `--name`, `--out`. A run writes `records.jsonl` as it goes, so re-running the same command resumes. `python -m memcheck report results/<run>` recomputes the summary.

## Plug in your own retriever

Write one JSON object per question. `qid` is `<sample_id>/<index of the question in that sample's qa list>`; `python -m memcheck export-questions --out questions.jsonl` lists all 1,986 with their text and category.

```json
{"qid": "conv-26/0", "memories": [{"text": "[D1:3 | 1:56 pm on 8 May, 2023] Caroline: I went to a LGBTQ support group yesterday...", "dia_id": "D1:3", "score": 0.82}]}
{"qid": "conv-26/1", "context": "free-text context, if your system returns one string"}
```

`dia_id` is optional, but with it the tool knows exactly whether the evidence turn was retrieved. Without it, it matches the evidence turn's text against your context (exact substring, or at least 80% of its content words), which works for raw turns and less well for rewritten memories.

```bash
python -m memcheck run --backend cli --contexts my_contexts.jsonl --memory-name my-retriever
```

**Example: FlyHash.** [`examples/flyhash_contexts.py`](examples/flyhash_contexts.py) builds contexts with the FlyHash and SimHash classes from [flyhash-agent-memory](https://github.com/mohanji/flyhash-agent-memory) (Bheemadolu & Odhiambo, [arXiv:2611.05791](https://arxiv.org/abs/2611.05791)), or with exact dense cosine as a reference, over MiniLM embeddings of the dialogue turns:

```bash
git clone https://github.com/mohanji/flyhash-agent-memory
pip install numpy sentence-transformers
python examples/flyhash_contexts.py --flyhash-src flyhash-agent-memory/src --method flyhash --active-bits 32 --out flyhash-k32.jsonl
python -m memcheck run --backend cli --contexts flyhash-k32.jsonl --memory-name flyhash-k32
```

Running the same questions with several code sizes, or against `--method dense`, shows whether shorter codes lead to more cases where both models agree on a wrong answer.

## What is measured

| Measure | Meaning |
|---|---|
| model A / B correct | judged correct (third-company LLM judge; LoCoMo F1 >= 0.5 if no judge) |
| agree | the two answers say the same thing and get the same grade |
| wrong when they agree | among agreeing pairs, share that are wrong. 1 minus this is the precision of "answer only when both agree" |
| wrong when they agree on a real answer | the same, leaving out pairs where both said "Not mentioned" (an honest non-answer is a different failure from a confident wrong answer) |
| both wrong, observed vs if independent | if the two models' errors were independent, both would be wrong on (1 - acc A)(1 - acc B) of questions |
| error correlation φ | correlation between "A wrong" and "B wrong". Two models with error correlation φ are worth 2/(1+φ) independent checks |
| retrieval full / incomplete | whether every evidence turn LoCoMo lists for the question was in the retrieved memory |

Categories follow the official scorer ([`task_eval/evaluation.py`](https://github.com/snap-research/locomo/blob/main/task_eval/evaluation.py)): 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial (the right answer is to say it was not mentioned). Proportions carry Wilson 95% intervals; φ and differences carry stratified bootstrap 95% intervals.

## First numbers

_Filled in from [`results/`](results/) after the first run._

## Limits of this first run

- 200 questions, 40 per category, one seed. Per-category intervals are wide (about ±15 points).
- One retriever (BM25 over raw turns, top 10), one pair of models, one judge. Other retrievers and models can move every number.
- The judge is an LLM. Its calls can be checked against LoCoMo's F1 rule, which is reported next to it in `records.jsonl`.
- LoCoMo's evidence labels are imperfect; a few ids are malformed and are dropped. "Incomplete retrieval" means LoCoMo's marked evidence was missing, not that the answer was impossible.
- The sample is balanced across categories, so pooled numbers are not LoCoMo-wide averages.
- This is an exploratory measurement, not a preregistered test.

## Try it yourself: Poly Duo

The same idea for everyday use: [Poly Duo](https://github.com/vadimchernets/poly-duo) ([polyhelper.ai/duo](https://polyhelper.ai/duo)) lets your own ChatGPT and Claude check each other on your phone, free and open source. It is a quick way to see two models agree, disagree, and sometimes agree on the same mistake.

## Related work by the author

- V. Chernets, *Agreement Is Not Independent Evidence: Auditable Multi-Model Synthesis Without an API*, preprint, SSRN 7390698, 2026. [polyhelper.ai/research/agreement-not-evidence](https://polyhelper.ai/research/agreement-not-evidence/), [doi:10.5281/zenodo.22821595](https://doi.org/10.5281/zenodo.22821595).
- V. Chernets, *Agreement Inversion Is Not a Mid-Tier Property: Precision of Agreed LLM Answers Follows Pair Accuracy*, preprint, SSRN 7682960, 2026. [polyhelper.ai/research/agreement-inversion](https://polyhelper.ai/research/agreement-inversion/), [doi:10.5281/zenodo.24090580](https://doi.org/10.5281/zenodo.24090580).

If you use this tool, please cite the repository (see [`CITATION.cff`](CITATION.cff)) and LoCoMo:

> Maharana, A., Lee, D.-H., Tulyakov, S., Bansal, M., Barbieri, F., & Fang, Y. (2024). Evaluating Very Long-Term Conversational Memory of LLM Agents. ACL 2024.

## Licenses

- **Code:** MIT ([`LICENSE`](LICENSE)).
- **LoCoMo data:** © the LoCoMo authors, [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (see the [LoCoMo repository](https://github.com/snap-research/locomo/blob/main/LICENSE.txt)). The dataset is **not** included here; `python -m memcheck download` fetches it from the official repository. Non-commercial terms apply to the data.
- **Files under `results/`** contain model answers to LoCoMo questions (question ids, not the questions or gold answers). As material derived from LoCoMo they are shared under CC BY-NC 4.0.
