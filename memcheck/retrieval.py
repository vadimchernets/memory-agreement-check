"""Memory retrieval: a plain BM25 'simple memory' over dialogue turns, or contexts supplied by
any external retriever as JSONL. Also measures whether the retrieved memory contains the
evidence turns LoCoMo marks for each question."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .locomo import Question, Turn

_TOK = re.compile(r"[a-z0-9]+")
STOP = set("""a an and are as at be been but by did do does for from had has have he her his how i if in
into is it its me my of on or our she so that the their them then there they this to was we were what
when where which who whom why will with would you your""".split())


def tokens(text: str) -> list[str]:
    return [t for t in _TOK.findall(text.lower()) if t not in STOP]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokens(d)) for d in docs]
        self.len = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokens(query)
        out = []
        for d, L in zip(self.docs, self.len):
            s = 0.0
            for t in q:
                f = d.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * L / (self.avg or 1)))
            out.append(s)
        return out

    def top(self, query: str, k: int) -> list[int]:
        sc = self.scores(query)
        order = sorted(range(len(sc)), key=lambda i: (-sc[i], i))
        return order[:k]


@dataclass
class Retrieved:
    context: str
    dia_ids: list[str] = field(default_factory=list)   # ids of retrieved turns when known
    source: str = "bm25"


class SimpleMemory:
    """BM25 over every turn of the conversation; top-k turns, shown in chronological order."""

    def __init__(self, convs: dict[str, list[Turn]], k: int = 10):
        self.convs, self.k = convs, k
        self._idx: dict[str, BM25] = {}

    def retrieve(self, q: Question) -> Retrieved:
        turns = self.convs[q.sample_id]
        if q.sample_id not in self._idx:
            self._idx[q.sample_id] = BM25([t.search_text() for t in turns])
        top = sorted(self._idx[q.sample_id].top(q.question, self.k))
        chosen = [turns[i] for i in top]
        return Retrieved(context="\n".join(t.render() for t in chosen), dia_ids=[t.dia_id for t in chosen],
                         source=f"bm25-top{self.k}")


class ExternalMemory:
    """Contexts produced by any other system, one JSON object per line:
       {"qid": "conv-26/0", "memories": [{"text": "...", "dia_id": "D1:3"}, ...]}
    or {"qid": "conv-26/0", "context": "free text"}. `dia_id` is optional but lets us measure
    retrieval exactly; without it we fall back to matching the evidence turns' text."""

    def __init__(self, path: Path, name: str = "external"):
        self.name = name
        self.by: dict[str, dict] = {}
        for line in Path(path).read_text().splitlines():
            if line.strip():
                o = json.loads(line)
                self.by[str(o["qid"])] = o

    def has(self, q: Question) -> bool:
        return q.qid in self.by

    def retrieve(self, q: Question) -> Retrieved:
        o = self.by.get(q.qid)
        if o is None:
            raise KeyError(f"no context for {q.qid} in the external JSONL")
        mems = o.get("memories")
        if mems:
            lines, ids = [], []
            for m in mems:
                if isinstance(m, str):
                    lines.append(m)
                    continue
                lines.append(str(m.get("text", "")))
                for k in ("dia_id", "dia_ids"):
                    v = m.get(k)
                    if v:
                        ids += [v] if isinstance(v, str) else list(v)
            return Retrieved(context="\n".join(lines), dia_ids=ids, source=self.name)
        return Retrieved(context=str(o.get("context", "")), dia_ids=list(o.get("dia_ids", []) or []), source=self.name)


def _norm(s: str) -> str:
    return " ".join(_TOK.findall(s.lower()))


def coverage(q: Question, r: Retrieved, turns_by_id: dict[str, Turn]) -> dict:
    """Did the retrieved memory contain the evidence turns? status: full | partial | none | unknown."""
    ev = [e for e in q.evidence if e in turns_by_id]
    if not ev:
        return {"status": "unknown", "found": 0, "needed": 0, "method": "none"}
    ids = set(r.dia_ids)
    method = "ids" if ids else "text"
    ctx_norm = _norm(r.context)
    ctx_tok = set(ctx_norm.split())
    found = 0
    for e in ev:
        if ids:
            hit = e in ids
        else:
            t = _norm(turns_by_id[e].text)
            toks = [x for x in t.split() if x not in STOP]
            hit = bool(t) and (t in ctx_norm or (len(toks) >= 3 and sum(x in ctx_tok for x in toks) / len(toks) >= 0.8))
        found += bool(hit)
    status = "full" if found == len(ev) else ("none" if found == 0 else "partial")
    return {"status": status, "found": found, "needed": len(ev), "method": method}
