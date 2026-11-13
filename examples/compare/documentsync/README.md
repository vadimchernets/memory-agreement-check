# Worked example: FP8 vs bf16 on a self-hosted RAG system

Contributed by Anagha Langhe ([DocumentSyncAI](https://github.com/ana-lan/DocumentSyncAI-RAG-based-QA-for-Documents)).

Qwen2.5-7B-Instruct served with vLLM 0.31.0 on one NVIDIA L4, once in bf16 and once in FP8. Both builds answered the same 120 questions over 3 research papers (8 question types, 15 each), and for each question both saw exactly the same 5 retrieved passages: retrieval ran once and was frozen.

## Files

- `run_bf16.jsonl`, `run_fp8.jsonl`: one line per question with `qid`, `answer`, `correct` and `category`. The answers were graded upstream (exact number rules plus an LLM judge), so each line carries its own `correct` field and no gold file is needed.
- `report.md`: the output of the command below

## Run it

```bash
python -m memcheck compare examples/compare/documentsync/run_bf16.jsonl examples/compare/documentsync/run_fp8.jsonl --label-a bf16 --label-b FP8
```

## What it shows

- 72 vs 71 correct, with 1 flip in bf16's favour and none the other way (exact McNemar p = 1.0)
- Accuracy difference +0.8 points, 95% interval about -2.3 to +4.6
- On the 48 questions both builds got wrong, they gave the same wrong answer 42 times

Full write-up: [RESULTS.md](https://github.com/ana-lan/DocumentSyncAI-RAG-based-QA-for-Documents/blob/main/benchmarks/RESULTS.md)