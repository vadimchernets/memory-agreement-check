from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import backends, locomo, report
from .retrieval import ExternalMemory, SimpleMemory


def _log_to(path: Path):
    def log(msg: str):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, file=sys.stderr, flush=True)
        with open(path, "a") as f:
            f.write(line + "\n")
    return log


def cmd_download(a):
    p = locomo.download(Path(a.data) if a.data else None, force=a.force)
    print(f"LoCoMo ready at {p} (CC BY-NC 4.0, snap-research/locomo @ {locomo.LOCOMO_COMMIT[:10]})")


def cmd_export(a):
    convs, qs = locomo.load(Path(a.data) if a.data else None)
    out = open(a.out, "w") if a.out else sys.stdout
    for q in qs:
        out.write(json.dumps({"qid": q.qid, "sample_id": q.sample_id, "question": q.question,
                              "category": q.category, "category_name": q.category_name}) + "\n")
    if a.out:
        out.close()
        print(f"{len(qs)} questions -> {a.out}", file=sys.stderr)


def cmd_run(a):
    data = Path(a.data) if a.data else locomo.default_data_path()
    if not data.exists():
        locomo.download(data)
    convs, qs = locomo.load(data)
    cats = [int(c) for c in a.categories.split(",")] if a.categories else None
    pa, pb, pj = backends.PRESETS[a.backend]
    spec_a, spec_b = a.model_a or pa, a.model_b or pb
    spec_j = a.judge or pj
    for s in (spec_a, spec_b):
        if s.startswith("api:") and s.count(":") < 2 or s.endswith(":"):
            sys.exit(f"{s!r}: name the model, e.g. --model-a api:openai:<model> "
                     "(or set MEMCHECK_OPENAI_MODEL / MEMCHECK_ANTHROPIC_MODEL)")
    workdir = str(Path(a.workdir or Path.cwd()).resolve())
    A = backends.make(spec_a, workdir, a.max_concurrent, a.effort)
    B = backends.make(spec_b, workdir, a.max_concurrent, a.effort)
    J = None if spec_j in ("rule", "none") else backends.make(spec_j, workdir, a.max_concurrent, a.effort)
    if A.family == B.family and A.family != "offline":
        print(f"warning: both models are from {A.family}; the point of the check is two different companies",
              file=sys.stderr)
    if a.contexts:
        memory = ExternalMemory(Path(a.contexts), name=a.memory_name or Path(a.contexts).stem)
        pool = [q for q in qs if memory.has(q)]
        print(f"external contexts cover {len(pool)} of {len(qs)} questions", file=sys.stderr)
        qs = pool
        mem_desc = f"external:{memory.name}"
    else:
        memory = SimpleMemory(convs, k=a.k)
        mem_desc = f"bm25-top{a.k}"
    chosen = locomo.sample(qs, a.n, a.seed, balanced=not a.random, categories=cats)
    name = a.name or f"{mem_desc}-n{len(chosen)}-seed{a.seed}"
    out = Path(a.out) / name
    out.mkdir(parents=True, exist_ok=True)
    from .run import run
    run(chosen, memory, convs, A, B, J, out, a.seed, a.workers,
        {"run_name": name, "memory": mem_desc, "balanced": not a.random, "k": a.k, "effort": a.effort,
         "categories": cats}, pace=a.pace, log=_log_to(out / "progress.log"))
    s = report.write(out, a.seed)
    print((out / "summary.md").read_text())
    return s


def cmd_report(a):
    report.write(Path(a.dir), a.seed)
    print((Path(a.dir) / "summary.md").read_text())


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m memcheck",
                                description="Does agreement between two LLMs from different companies still mean "
                                            "anything when both read the same retrieved memory? Measured on LoCoMo.")
    p.add_argument("--data", help="path to locomo10.json (default ~/.cache/memory-agreement-check/locomo10.json)")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="fetch LoCoMo (CC BY-NC 4.0) at the pinned commit and check its hash")
    d.add_argument("--force", action="store_true")
    d.set_defaults(f=cmd_download)

    e = sub.add_parser("export-questions", help="write every LoCoMo question with its qid, for external retrievers")
    e.add_argument("--out")
    e.set_defaults(f=cmd_export)

    r = sub.add_parser("run", help="retrieve, ask two models separately, judge, summarise")
    r.add_argument("--backend", choices=sorted(backends.PRESETS), default="dry",
                   help="preset for model A / model B / judge: cli (subscriptions), api (keys), dry (offline)")
    r.add_argument("--model-a", help="override, e.g. cli:codex:gpt-5.5, api:openai:<model>, dry:oracle")
    r.add_argument("--model-b", help="override, e.g. cli:claude:opus, api:anthropic:<model>")
    r.add_argument("--judge", help="third-company judge, e.g. cli:kimi, api:openai:<model>; 'rule' = LoCoMo F1 only")
    r.add_argument("--contexts", help="JSONL from your own retriever (see README); default: built-in BM25 memory")
    r.add_argument("--memory-name", help="label for the external retriever in the results")
    r.add_argument("--k", type=int, default=10, help="turns retrieved by the built-in BM25 memory (default 10)")
    r.add_argument("--n", type=int, default=200, help="questions to sample; 0 = all (default 200)")
    r.add_argument("--seed", type=int, default=2026)
    r.add_argument("--random", action="store_true", help="simple random sample instead of equal per category")
    r.add_argument("--categories", help="comma-separated LoCoMo category ids to keep, e.g. 1,2,3,4")
    r.add_argument("--workers", type=int, default=2, help="questions in flight at once (default 2)")
    r.add_argument("--max-concurrent", type=int, default=2, help="max parallel calls per CLI tool (default 2)")
    r.add_argument("--pace", type=float, default=0.0, help="seconds to wait after each question (rate limits)")
    r.add_argument("--effort", default="low", help="codex reasoning effort (default low)")
    r.add_argument("--workdir", help="folder the CLI agents are started in (read-only); default: current folder")
    r.add_argument("--out", default="results", help="results folder (default results/)")
    r.add_argument("--name", help="run name (subfolder of --out)")
    r.set_defaults(f=cmd_run)

    s = sub.add_parser("report", help="recompute summary.md / summary.json for a run folder")
    s.add_argument("dir")
    s.add_argument("--seed", type=int, default=2026)
    s.set_defaults(f=cmd_report)

    a = p.parse_args(argv)
    return a.f(a)
