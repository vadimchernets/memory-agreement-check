"""LoCoMo loading, download and sampling.

The dataset is NOT shipped with this repository. `download()` fetches it from the
official snap-research/locomo repository at a pinned commit and checks its hash.
LoCoMo is licensed CC BY-NC 4.0 by its authors (Maharana et al., ACL 2024).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

LOCOMO_COMMIT = "cbfbc1dba6bc53d00625212a0f22d55ffee7c1fc"
LOCOMO_URL = f"https://raw.githubusercontent.com/snap-research/locomo/{LOCOMO_COMMIT}/data/locomo10.json"
LOCOMO_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"

# Category ids as used by the official scorer (snap-research/locomo, task_eval/evaluation.py):
# category 1 is scored as multi-hop (answer split on commas), 5 as adversarial (abstention),
# 2 holds the "When did ..." questions, 3 the open-domain/commonsense ones, 4 single-hop facts.
CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
ABSTAIN_GOLD = "Not mentioned (the conversation does not contain this information)"

_EVID = re.compile(r"D(\d+):(\d+)")


def default_data_path() -> Path:
    env = os.environ.get("MEMCHECK_DATA")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "memory-agreement-check" / "locomo10.json"


def download(path: Path | None = None, force: bool = False) -> Path:
    path = Path(path or default_data_path())
    if path.exists() and not force:
        _check_hash(path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(LOCOMO_URL, timeout=120) as r, open(tmp, "wb") as f:
        f.write(r.read())
    _check_hash(tmp)
    tmp.replace(path)
    return path


def _check_hash(path: Path) -> None:
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if h != LOCOMO_SHA256:
        raise SystemExit(f"{path}: sha256 {h} does not match the pinned LoCoMo file {LOCOMO_SHA256}. "
                         "Delete it and run `python -m memcheck download` again.")


@dataclass
class Turn:
    dia_id: str
    session: int
    turn: int
    date: str
    speaker: str
    text: str
    caption: str = ""

    def render(self) -> str:
        s = f"[{self.dia_id} | {self.date}] {self.speaker}: {self.text}"
        if self.caption:
            s += f" [shared an image: {self.caption}]"
        return s

    def search_text(self) -> str:
        return f"{self.speaker} {self.text} {self.caption}"


@dataclass
class Question:
    qid: str
    sample_id: str
    index: int
    question: str
    answer: str
    category: int
    evidence: list[str] = field(default_factory=list)

    @property
    def category_name(self) -> str:
        return CATEGORIES.get(self.category, str(self.category))

    @property
    def adversarial(self) -> bool:
        return self.category == 5


def parse_evidence(raw) -> list[str]:
    """Normalise LoCoMo evidence ids. A few entries are malformed ('D8:6; D9:17', 'D:11:26');
    we keep every well-formed Dn:m found and drop the rest."""
    out: list[str] = []
    for e in raw or []:
        for s, t in _EVID.findall(str(e)):
            k = f"D{int(s)}:{int(t)}"
            if k not in out:
                out.append(k)
    return out


def load(path: Path | None = None):
    """Return (conversations, questions). conversations: sample_id -> chronological list of Turn."""
    data = json.loads(Path(path or default_data_path()).read_text())
    convs: dict[str, list[Turn]] = {}
    questions: list[Question] = []
    for sample in data:
        sid = sample["sample_id"]
        conv = sample["conversation"]
        turns: list[Turn] = []
        for key, val in conv.items():
            m = re.fullmatch(r"session_(\d+)", key)
            if not m or not isinstance(val, list):
                continue
            sess = int(m.group(1))
            date = conv.get(f"session_{sess}_date_time", "")
            for t in val:
                ids = _EVID.findall(t.get("dia_id", ""))
                tn = int(ids[0][1]) if ids else len(turns)
                turns.append(Turn(dia_id=f"D{sess}:{tn}", session=sess, turn=tn, date=date,
                                  speaker=t.get("speaker", ""), text=t.get("text", ""),
                                  caption=t.get("blip_caption", "") or ""))
        turns.sort(key=lambda x: (x.session, x.turn))
        convs[sid] = turns
        for i, qa in enumerate(sample["qa"]):
            cat = int(qa["category"])
            ans = ABSTAIN_GOLD if cat == 5 else str(qa.get("answer", ""))
            questions.append(Question(qid=f"{sid}/{i}", sample_id=sid, index=i, question=qa["question"],
                                      answer=ans, category=cat, evidence=parse_evidence(qa.get("evidence"))))
    return convs, questions


def sample(questions: list[Question], n: int, seed: int, balanced: bool = True,
           categories: list[int] | None = None) -> list[Question]:
    """Seeded sample. balanced=True draws (about) n/len(categories) per category."""
    rng = random.Random(seed)
    pool = [q for q in questions if categories is None or q.category in categories]
    if n <= 0 or n >= len(pool):
        return sorted(pool, key=lambda q: (q.sample_id, q.index))
    if not balanced:
        return sorted(rng.sample(pool, n), key=lambda q: (q.sample_id, q.index))
    cats = sorted({q.category for q in pool})
    by = {c: sorted([q for q in pool if q.category == c], key=lambda q: (q.sample_id, q.index)) for c in cats}
    quota = {c: n // len(cats) for c in cats}
    for c in cats[: n - sum(quota.values())]:
        quota[c] += 1
    picked: list[Question] = []
    spare = 0
    for c in cats:
        take = min(quota[c], len(by[c]))
        spare += quota[c] - take
        picked += rng.sample(by[c], take)
    if spare:  # a category was too small: fill from the rest at random
        rest = [q for q in pool if q not in picked]
        picked += rng.sample(rest, spare)
    return sorted(picked, key=lambda q: (q.sample_id, q.index))
