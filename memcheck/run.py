"""The experiment loop: retrieve memory -> ask two models separately -> judge -> one JSON record per question.
Records are appended as they finish, so an interrupted run resumes where it stopped."""
from __future__ import annotations

import hashlib
import json
import platform
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import __version__, prompts, scoring
from .backends import Backend, BackendError
from .locomo import LOCOMO_COMMIT, LOCOMO_SHA256, Question
from .retrieval import coverage

_lock = threading.Lock()
_JSON = re.compile(r"\{[^{}]*\}", re.S)


def parse_judge(text: str):
    for m in _JSON.finditer(text or ""):
        try:
            o = json.loads(m.group(0))
        except ValueError:
            continue
        if all(k in o for k in ("answer_1_correct", "answer_2_correct", "same")):
            return {k: bool(o[k]) for k in ("answer_1_correct", "answer_2_correct", "same")}
    return None


def _order(seed: int, qid: str) -> str:
    h = int(hashlib.sha256(f"{seed}|{qid}".encode()).hexdigest()[:8], 16)
    return "ab" if h % 2 == 0 else "ba"


def one(q: Question, memory, turns_by_id, a: Backend, b: Backend, judge: Backend | None, seed: int) -> dict:
    r = memory.retrieve(q)
    cov = coverage(q, r, turns_by_id[q.sample_id])
    prompt = prompts.ANSWER.format(context=r.context or "(no memories retrieved)", question=q.question)
    item = {"qid": q.qid, "gold": q.answer, "coverage": cov["status"]}
    with ThreadPoolExecutor(2) as ex:   # the two models answer at the same time and never see each other
        fa, fb = ex.submit(a.ask, prompt, item), ex.submit(b.ask, prompt, item)
        ra, rb = fa.result(), fb.result()
    rule = {
        "a_score": round(scoring.locomo_score(ra.text, q.answer, q.category), 4),
        "b_score": round(scoring.locomo_score(rb.text, q.answer, q.category), 4),
        "a_correct": scoring.rule_correct(ra.text, q.answer, q.category),
        "b_correct": scoring.rule_correct(rb.text, q.answer, q.category),
        "same": scoring.rule_same(ra.text, rb.text),
    }
    jrec = None
    if judge is not None:
        order = _order(seed, q.qid)
        a1, a2 = (ra.text, rb.text) if order == "ab" else (rb.text, ra.text)
        jp = prompts.JUDGE.format(question=q.question, gold=q.answer, a1=a1, a2=a2)
        parsed, raw, jmodel = None, "", judge.spec
        for _ in range(2):
            rj = judge.ask(jp, {"qid": q.qid})
            raw, jmodel = rj.text, rj.model
            parsed = parse_judge(rj.text)
            if parsed:
                break
        if parsed:
            c1, c2 = parsed["answer_1_correct"], parsed["answer_2_correct"]
            jrec = {"spec": judge.spec, "model": jmodel, "order": order,
                    "a_correct": c1 if order == "ab" else c2, "b_correct": c2 if order == "ab" else c1,
                    "same": parsed["same"]}
        else:
            jrec = {"spec": judge.spec, "model": jmodel, "order": order, "failed": True, "raw": raw[:500]}
    if jrec and not jrec.get("failed"):
        ac, bc, same, src = jrec["a_correct"], jrec["b_correct"], jrec["same"], "judge"
    else:
        ac, bc, same, src = rule["a_correct"], rule["b_correct"], rule["same"], "rule"
    # "agree" = same answer AND the grader treats them alike; a judge saying "same" while marking one
    # right and one wrong is inconsistent, and we do not count that pair as agreeing.
    agree = bool(same and ac == bc)
    return {
        "qid": q.qid, "category": q.category, "category_name": q.category_name,
        "retrieval": {"source": r.source, "dia_ids": r.dia_ids, "context_chars": len(r.context), **cov},
        "a": {"spec": a.spec, "family": a.family, "model": ra.model, "text": ra.text, "seconds": ra.seconds,
              "tool_used": ra.tool_used, "abstain": scoring.is_abstain(ra.text)},
        "b": {"spec": b.spec, "family": b.family, "model": rb.model, "text": rb.text, "seconds": rb.seconds,
              "tool_used": rb.tool_used, "abstain": scoring.is_abstain(rb.text)},
        "rule": rule, "judge": jrec,
        "final": {"source": src, "a_correct": ac, "b_correct": bc, "same": same, "agree": agree,
                  "judge_inconsistent": bool(same and ac != bc)},
    }


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def run(questions: list[Question], memory, convs, a: Backend, b: Backend, judge: Backend | None,
        out: Path, seed: int, workers: int, config: dict, pace: float = 0.0, log=print) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    rec_path, err_path = out / "records.jsonl", out / "errors.jsonl"
    done = set()
    if rec_path.exists():
        for line in rec_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["qid"])
    meta = {"tool": "memory-agreement-check", "version": __version__, "git": _git_rev(),
            "python": platform.python_version(), "stemmer": scoring.STEMMER,
            "locomo_commit": LOCOMO_COMMIT, "locomo_sha256": LOCOMO_SHA256,
            "model_a": a.spec, "model_b": b.spec, "judge": judge.spec if judge else "rule (LoCoMo F1 >= 0.5)",
            "n_questions": len(questions), "seed": seed, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "qids": [q.qid for q in questions], **config}
    meta_path = out / "meta.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps(meta, indent=1))
    todo = [q for q in questions if q.qid not in done]
    log(f"{len(done)} already done, {len(todo)} to go -> {out}")
    turns_by_id = {sid: {t.dia_id: t for t in ts} for sid, ts in convs.items()}
    rng = random.Random(seed)
    rng.shuffle(todo)   # spread categories over time, so a stop half-way leaves a mixed sample
    counter = {"n": 0, "err": 0}

    def work(q: Question):
        try:
            rec = one(q, memory, turns_by_id, a, b, judge, seed)
            with _lock:
                with open(rec_path, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counter["n"] += 1
                f_ = rec["final"]
                log(f"[{counter['n']}/{len(todo)}] {q.qid} {q.category_name:11s} mem={rec['retrieval']['status']:7s} "
                    f"A={'ok' if f_['a_correct'] else 'x '} B={'ok' if f_['b_correct'] else 'x '} "
                    f"agree={'y' if f_['agree'] else 'n'}")
        except (BackendError, KeyError) as e:
            with _lock:
                counter["err"] += 1
                with open(err_path, "a") as f:
                    f.write(json.dumps({"qid": q.qid, "error": str(e)[:1000], "at": time.time()}) + "\n")
                log(f"ERROR {q.qid}: {str(e)[:200]}")
        if pace:
            time.sleep(pace)

    with ThreadPoolExecutor(max(1, workers)) as ex:
        list(ex.map(work, todo))
    log(f"finished: {counter['n']} new records, {counter['err']} errors")
    return rec_path
