"""LoCoMo-style answer scoring (token F1 after the official normalisation) plus helpers
that turn F1 into a correct/incorrect call when no LLM judge is used."""
from __future__ import annotations

import re
import string
from collections import Counter

try:  # the official scorer stems with NLTK's Porter stemmer; we use it when available
    from nltk.stem import PorterStemmer  # type: ignore
    _ps = PorterStemmer()
    STEMMER = "nltk-porter"

    def _stem(w: str) -> str:
        return _ps.stem(w)
except Exception:  # pragma: no cover - depends on environment
    STEMMER = "none"

    def _stem(w: str) -> str:
        return w

ABSTAIN_MARKERS = ("not mentioned", "no information available")
_ART = re.compile(r"\b(a|an|the|and)\b")


def normalize_answer(s: str) -> str:
    s = str(s).replace(",", "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ART.sub(" ", s)
    return " ".join(s.split())


def f1_score(pred: str, gold: str) -> float:
    p = [_stem(w) for w in normalize_answer(pred).split()]
    g = [_stem(w) for w in normalize_answer(gold).split()]
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def f1_multi(pred: str, gold: str) -> float:
    preds = [x.strip() for x in str(pred).split(",")]
    golds = [x.strip() for x in str(gold).split(",")]
    return sum(max(f1_score(p, g) for p in preds) for g in golds) / len(golds)


_ABSTAIN_STRIP = re.compile(
    r"not mentioned|no information available|not (?:stated|specified|given|available|provided|in the excerpts)|"
    r"cannot be determined|can't be determined|unknown|the excerpts? (?:do|does) not (?:say|mention|contain|state)|"
    r"in the (?:excerpts|conversation|memory)|\\b(?:it|this|that|is|was|are|the|a|answer)\\b", re.I)


def is_abstain(ans: str) -> bool:
    """An abstention leads with 'Not mentioned' or says nothing else of substance.
    'When Jolene was 10 (the exact year is not mentioned)' is NOT an abstention."""
    a = str(ans).lower().strip()
    if not any(m in a for m in ABSTAIN_MARKERS):
        return False
    if re.match(r"^\W*(not mentioned|no information available)\b", a):   # leads with the abstention
        return True
    rest = re.findall(r"[a-z0-9]+", _ABSTAIN_STRIP.sub(" ", a))
    return len(rest) <= 2


def locomo_score(pred: str, gold: str, category: int) -> float:
    """Same rules as snap-research/locomo task_eval/evaluation.py::eval_question_answering."""
    if category == 5:
        return 1.0 if is_abstain(pred) else 0.0
    if category == 3:
        gold = str(gold).split(";")[0].strip()
    if category == 1:
        return f1_multi(pred, gold)
    return f1_score(pred, gold)


def rule_correct(pred: str, gold: str, category: int, threshold: float = 0.5) -> bool:
    if category != 5 and is_abstain(pred):
        return False
    return locomo_score(pred, gold, category) >= threshold


def rule_same(a: str, b: str, threshold: float = 0.5) -> bool:
    """Do two answers say the same thing? Both abstain, or token F1 between them >= threshold."""
    xa, xb = is_abstain(a), is_abstain(b)
    if xa or xb:
        return xa and xb
    return f1_score(a, b) >= threshold
