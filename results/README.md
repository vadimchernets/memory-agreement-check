# Results

## Run `locomo-bm25top10-n200-seed2026` (11 November 2026)

- **Questions:** 200 LoCoMo questions, 40 from each category, seed 2026 (ids in `meta.json`).
- **Memory:** built-in BM25 over the raw dialogue turns, top 10 turns, shown in chronological order.
- **Model A:** `gpt-6.1-sol` (OpenAI) through the Codex CLI 0.166.0, reasoning effort low, read-only sandbox.
- **Model B:** `claude-sonnet-5-5` (Anthropic) through Claude Code 2.1.322, all tools disabled.
- **Judge:** `kimi-for-coding` (Moonshot) through the Kimi CLI 2.4.1. It was blind to which model gave which answer, and the order was randomised per question.
- Neither answering model used a tool on any question. 0 errors.
- Full tables in both grading views, including the per-category split: [`summary.md`](locomo-bm25top10-n200-seed2026/summary.md). Every answer and grade: `records.jsonl` (question ids only; join with LoCoMo for the text).

### What we see

Numbers are for categories 1-4: 160 questions, of which 156 have usable evidence ids. Judge view unless noted; 95% intervals in brackets.

1. **With full memory, agreement on a real answer was a good signal.** In 71 questions where every evidence turn was retrieved, both models gave the same real answer on 63, and 2 of those were wrong (3.2% [0.9, 10.9]).
2. **With incomplete memory, it was close to a coin flip.** In 85 questions where at least one evidence turn was missing, they gave the same real answer on 33, and 15 were wrong (45.5% [29.8, 62.0]). The difference is +42.3 points [+24.3, +59.3] pooled and +44.4 points [+20.7, +64.6] as the average of within-category differences.
3. **The F1 rule gives the same picture:** 12.3% (7/57) wrong with full memory against 65.5% (19/29) with incomplete memory.
4. **Agreeing on "Not mentioned" is most of the raw count.** Counting pairs where both said "Not mentioned", agreement was wrong in 4.7% (3/64) of cases with full memory and 73.5% (50/68) with incomplete memory. Of the 54 agreed-and-wrong pairs, 37 were both models declining as the prompt instructs, which is a different and milder failure than a shared wrong answer.
5. **Errors were strongly correlated.** φ = 0.81 [0.72, 0.88] between "GPT wrong" and "Claude wrong". Both were wrong on 40.0% of questions, against 20.1% expected if their errors were independent.
6. **By category:** agreement on a real answer was least reliable on multi-hop questions (wrong 9 of 22 times) and most reliable on temporal (3 of 31) and single-hop (3 of 30). On adversarial questions both models usually declined correctly (92.5% and 85.0%), and they gave the same trap answer twice in 40.

### How far this goes

- An association, not a cause. Retrieval quality goes with question type: evidence was fully retrieved for 15% of multi-hop and 15% of open-domain questions against 68% of temporal and 80% of single-hop. The within-category table in `summary.md` points the same way, but several of its cells are small. A causal version would answer the same questions with and without the evidence turns.
- "Incomplete" means LoCoMo's marked evidence was missing. For multi-hop questions a partial context can still allow a right answer, and LoCoMo's labels have known errors. Four questions with unparseable evidence ids are left out of the split: conv-50/39, conv-50/42, conv-26/30, conv-26/46.
- The grade depends on the grader. The judge and the F1 rule agree on 182/200 (model A) and 171/200 (model B) answers. The judge's raw replies were not saved in this run (the tool now saves them and reads only true/false values), so the judge view cannot be re-audited beyond its parsed grades. The F1-rule view needs no judge and supports the same conclusion.
- One pair of models, one retriever, one judge, one seed. The sample has equal numbers per category, so pooled rates are not LoCoMo-wide averages.
- Exploratory, not preregistered.

### Changes after review

One review by another AI family (Kimi) before release led to these changes, all recomputed from the same records with no new model calls: stricter detection of "Not mentioned" answers (an answer such as "when she was 10; the exact year is not mentioned" is no longer treated as declining); agreement defined by the judge's "same" call alone; strict parsing of the judge's true/false values; a second grading view by the F1 rule; within-category comparisons; real-answer agreement as the headline.
