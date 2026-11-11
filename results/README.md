# Results

## Run `locomo-bm25top10-n200-seed2026` (11 November 2026)

- **Questions:** 200 LoCoMo questions, 40 from each category, seed 2026 (ids in `meta.json`).
- **Memory:** built-in BM25 over the raw dialogue turns, top 10 turns, shown in chronological order.
- **Model A:** `gpt-6.1-sol` (OpenAI) through the Codex CLI, reasoning effort low, read-only sandbox.
- **Model B:** `claude-sonnet-5-5` (Anthropic) through the Claude Code CLI, all tools disabled.
- **Judge:** `kimi-for-coding` (Moonshot) through the Kimi CLI, blind to which model gave which answer (order randomised per question).
- **No tool use** was recorded for either answering model on any question. 0 errors, 0 retries needed.
- Full tables: [`summary.md`](locomo-bm25top10-n200-seed2026/summary.md); every answer and grade: `records.jsonl`.

### What we see

Numbers are for categories 1-4 (160 questions) unless noted. Brackets are 95% intervals.

1. **When memory held the evidence, agreement was a good signal.** In 71 questions where every evidence turn was retrieved, the two models agreed on 64, and 3 of those 64 were wrong (4.7% [1.6, 12.9]).
2. **When memory missed evidence, agreement was mostly wrong.** In 85 questions where at least one evidence turn was missing, they agreed on 67, and 50 of those were wrong (74.6% [63.1, 83.5]). The difference is +69.9 points [+58.2, +80.7].
3. **Most of those were honest "Not mentioned" answers, but not all.** Leaving out pairs where both said "Not mentioned", the models agreed on a substantive answer in 32 incomplete-memory questions and 15 of those were wrong (46.9% [30.9, 63.6]), against 2 of 63 (3.2%) with full memory.
4. **Agreed errors came with poor retrieval.** Of the 54 agreed-but-wrong answers, 50 (92.6% [82.4, 97.1]) had incomplete retrieval. Counting only substantive answers: 15 of 17.
5. **The two models' errors were strongly correlated.** Error correlation φ = 0.81 [0.72, 0.88]. Both were wrong on 40.0% of questions, against 20.1% expected if their errors were independent. With φ = 0.81 the pair is worth about 2/(1+0.81) ≈ 1.1 independent checks.
6. **By category,** agreement was least reliable on multi-hop questions (wrong in 22 of 35 agreements) and open-domain questions (17 of 29), and most reliable on single-hop (7 of 33) and temporal (8 of 36). On adversarial questions both models usually declined correctly (92.5% and 85.0%), and they agreed on the trap answer twice out of 40.

### How far this goes

- This shows an association, not a cause. Questions whose evidence BM25 misses may also be harder in other ways. A causal version would give the same questions with and without the evidence turns (for example, by removing them from an otherwise good context).
- "Incomplete" means LoCoMo's marked evidence was missing. For multi-hop questions with several evidence turns, a partial context can still allow a right answer, and LoCoMo's labels have known errors.
- One pair of models, one retriever, one judge, one seed. The LLM judge agreed with LoCoMo's F1 >= 0.5 rule on 182/200 (model A) and 171/200 (model B) answers. Most disagreements are paraphrases the F1 rule misses. The judge marked one pair "same" with different grades; that pair is not counted as agreeing.
- The sample has equal numbers per category, so pooled rates are not LoCoMo-wide averages.
- Exploratory, not preregistered.
