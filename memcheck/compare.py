"""Compare two runs on the same questions: two models, or two versions of one model (e.g. FP8 vs bf16).

Paired analysis: the 2x2 table of right/wrong, the discordant pairs, the exact McNemar test, a Tango
score interval for the accuracy difference, how many answers are word-for-word identical, how often the two runs
agree and are both wrong, and how many questions a test would need to detect a small drop.
No dependencies beyond the standard library."""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import scoring, stats

QID_KEYS = ("qid", "id", "question_id")
ANSWER_KEYS = ("answer", "prediction", "response", "output")
MATCHES = ("exact", "contains", "f1")


def _first(row: dict, keys):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def load_rows(path: Path) -> list[dict]:
    """JSONL (one object per line) or a JSON array."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    out = []
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise ValueError(f"{path}: row {i + 1} is not a JSON object")
        q = _first(r, QID_KEYS)
        if q is None:
            raise ValueError(f"{path}: row {i + 1} has no qid (accepted keys: {', '.join(QID_KEYS)})")
        out.append({**r, "_qid": str(q)})
    return out


def _by_qid(rows: list[dict], path) -> dict:
    d = {}
    for r in rows:
        if r["_qid"] in d:
            raise ValueError(f"{path}: duplicate qid {r['_qid']!r}")
        d[r["_qid"]] = r
    return d


def _as_bool(v, where: str) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str) and v.strip().lower() in ("true", "false", "1", "0", "yes", "no"):
        return v.strip().lower() in ("true", "1", "yes")
    raise ValueError(f"{where}: 'correct' must be true/false or 1/0, got {v!r}")


def grade(answer, gold, match: str = "exact", threshold: float = 0.5) -> bool:
    """Is `answer` right? `gold` may be one string or a list of acceptable answers.
    exact: equal after LoCoMo/SQuAD normalisation (case, punctuation, articles, spaces);
    contains: the normalised gold appears inside the normalised answer (whole words);
    f1: token F1 against the gold >= threshold."""
    if answer is None:
        return False
    golds = gold if isinstance(gold, list) else [gold]
    na = scoring.normalize_answer(answer)
    for g in golds:
        ng = scoring.normalize_answer(g)
        if match == "exact" and na == ng:
            return True
        if match == "contains" and ng and f" {ng} " in f" {na} ":
            return True
        if match == "f1" and scoring.f1_score(answer, g) >= threshold:
            return True
    return False


def pair_runs(rows_a, rows_b, gold_rows=None, match="exact", threshold=0.5, paths=("A", "B", "gold")):
    """Join the two runs on qid and decide right/wrong for each answer.
    With gold rows, answers are graded against them; otherwise each row needs a 'correct' field."""
    A, B = _by_qid(rows_a, paths[0]), _by_qid(rows_b, paths[1])
    G = _by_qid(gold_rows, paths[2]) if gold_rows is not None else None
    common = [q for q in A if q in B and (G is None or q in G)]
    pairs = []
    for q in common:
        ra, rb = A[q], B[q]
        rec = {"qid": q, "answer_a": _first(ra, ANSWER_KEYS), "answer_b": _first(rb, ANSWER_KEYS)}
        if G is not None:
            g = _first(G[q], ANSWER_KEYS + ("gold",))
            if g is None:
                raise ValueError(f"{paths[2]}: qid {q!r} has no gold answer (key 'answer')")
            for side, path in (("a", paths[0]), ("b", paths[1])):
                if rec[f"answer_{side}"] is None:
                    raise ValueError(f"{path}: qid {q!r} has no answer (keys: {', '.join(ANSWER_KEYS)}); "
                                     "with --gold every row needs one")
            rec["correct_a"] = grade(rec["answer_a"], g, match, threshold)
            rec["correct_b"] = grade(rec["answer_b"], g, match, threshold)
        else:
            for side, r, p in (("a", ra, paths[0]), ("b", rb, paths[1])):
                if "correct" not in r:
                    raise ValueError(f"{p}: qid {q!r} has no 'correct' field; pass --gold to grade answers")
                rec[f"correct_{side}"] = _as_bool(r["correct"], f"{p} qid {q!r}")
        found = {str(r["category"]) for r in ((G or {}).get(q), ra, rb) if r and r.get("category") is not None}
        if len(found) > 1:
            raise ValueError(f"qid {q!r}: the files give different categories {sorted(found)}")
        rec["category"] = found.pop() if found else None
        pairs.append(rec)
    unmatched = {"only_in_a": len([q for q in A if q not in B]), "only_in_b": len([q for q in B if q not in A]),
                 "missing_gold": 0 if G is None else len([q for q in A if q in B and q not in G])}
    return pairs, unmatched


def _table(pairs):
    n11 = sum(1 for p in pairs if p["correct_a"] and p["correct_b"])
    n10 = sum(1 for p in pairs if p["correct_a"] and not p["correct_b"])
    n01 = sum(1 for p in pairs if not p["correct_a"] and p["correct_b"])
    return n11, n10, n01, len(pairs) - n11 - n10 - n01


def _prop(k, n):
    p, lo, hi = stats.wilson(k, n)
    return {"k": k, "n": n, "p": p, "lo": lo, "hi": hi}


def _paired(pairs):
    n11, n10, n01, n00 = _table(pairs)
    n = len(pairs)
    d, lo, hi = stats.paired_diff_ci(n11, n10, n01, n00)
    return {"n": n, "acc_a": _prop(n11 + n10, n), "acc_b": _prop(n11 + n01, n),
            "both_right": n11, "only_a": n10, "only_b": n01, "both_wrong": n00,
            "b": n10, "c": n01, "discordant": n10 + n01,
            "mcnemar_exact_p": stats.mcnemar_exact(n10, n01), "diff": d, "diff_lo": lo, "diff_hi": hi}


def analyze(pairs, detect_points=(1.0,), alpha=0.05, power=0.8):
    if not pairs:
        raise ValueError("no questions in common between the two runs")
    out = _paired(pairs)
    n = out["n"]

    # agreement between the two answers (needs answer text)
    with_text = [p for p in pairs if p["answer_a"] is not None and p["answer_b"] is not None]
    if with_text:
        ident = [p for p in with_text if str(p["answer_a"]).strip() == str(p["answer_b"]).strip()]
        ident_norm = [p for p in with_text
                      if scoring.normalize_answer(p["answer_a"]) == scoring.normalize_answer(p["answer_b"])]
        same = [p for p in with_text if scoring.rule_same(str(p["answer_a"]), str(p["answer_b"]))]
        bw = [p for p in with_text if not p["correct_a"] and not p["correct_b"]]
        bw_same = [p for p in bw if scoring.rule_same(str(p["answer_a"]), str(p["answer_b"]))]
        bw = len(bw)
        out["answers"] = {
            "n_with_text": len(with_text),
            "identical": _prop(len(ident), len(with_text)),
            "identical_normalised": _prop(len(ident_norm), len(with_text)),
            "same": _prop(len(same), len(with_text)),
            "wrong_when_identical": _prop(sum(1 for p in ident if not p["correct_a"] and not p["correct_b"]), len(ident)),
            "wrong_when_same": _prop(sum(1 for p in same if not p["correct_a"] and not p["correct_b"]), len(same)),
            "same_among_both_wrong": _prop(len(bw_same), bw),
            "inconsistent_grades_when_identical": sum(1 for p in ident if p["correct_a"] != p["correct_b"]),
        }

    # how many questions would be needed to detect a small drop of B relative to A
    psi_obs = out["discordant"] / n
    _, _, psi_hi = stats.wilson(out["discordant"], n)
    plans = []
    for pts in detect_points:
        dlt = pts / 100
        row = {"points": pts, "best_case": stats.n_for_drop(dlt, dlt, alpha, power),
               "at_observed_rate": stats.n_for_drop(dlt, psi_obs, alpha, power) if psi_obs >= dlt else None,
               "at_upper_rate": stats.n_for_drop(dlt, psi_hi, alpha, power) if psi_hi >= dlt else None}
        plans.append(row)
    out["detect"] = {"alpha": alpha, "power": power, "flip_rate": psi_obs, "flip_rate_hi": psi_hi,
                     "min_flips_for_significance": _min_flips(alpha), "plans": plans}

    cats = sorted({p["category"] for p in pairs if p["category"] is not None},
                  key=lambda c: (not c.lstrip("-").isdigit(), int(c) if c.lstrip("-").isdigit() else 0, c))
    if cats:
        out["by_category"] = {c: _paired([p for p in pairs if p["category"] == c]) for c in cats}
        none = [p for p in pairs if p["category"] is None]
        if none:
            out["by_category"]["(no category)"] = _paired(none)
    out["flipped"] = {"only_a": [p["qid"] for p in pairs if p["correct_a"] and not p["correct_b"]],
                      "only_b": [p["qid"] for p in pairs if not p["correct_a"] and p["correct_b"]]}
    return out


def _min_flips(alpha):
    """Fewest discordant pairs at which the exact test can reach p <= alpha at all (all flips one way)."""
    m = 1
    while stats.mcnemar_exact(m, 0) > alpha:
        m += 1
    return m


def _cell(x) -> str:
    """Safe text for a Markdown table cell or list item."""
    return str(x).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _pct(x, digits=1):
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def _pts(x):
    return "n/a" if x is None else f"{100 * x:+.1f}"


def _ci(pr):
    if pr["n"] == 0:
        return f"0/0"
    return f"{pr['k']}/{pr['n']} = {_pct(pr['p'])} [{_pct(pr['lo'])}, {_pct(pr['hi'])}]"


def _pval(p):
    return "1.0" if p >= 0.9995 else (f"{p:.3f}" if p >= 0.001 else f"{p:.1e}")


def _fmt_n(n):
    return "more than 1,000,000" if n is None else f"{n:,}"


def verdict(s, la="A", lb="B") -> list[str]:
    """Plain-language reading of the result."""
    n, b, c, p = s["n"], s["b"], s["c"], s["mcnemar_exact_p"]
    m = b + c
    lines = []
    flips = f"{m} flip{'s' if m != 1 else ''} out of {n}"
    det = s["detect"]
    plan = det["plans"][0] if det["plans"] else None
    need = ""
    if plan:
        need = (f"; you need N ≈ {_fmt_n(plan['best_case'])} questions to detect a {plan['points']:g}-point drop"
                f" ({det['power']:.0%} power, exact McNemar, α = {det['alpha']:g}) if every flip goes the same way"
                f" (the fewest possible)")
        if plan["at_upper_rate"] is not None:
            need += (f", and about {_fmt_n(plan['at_upper_rate'])} if {_pct(det['flip_rate_hi'])} of answers flip"
                     f" (the upper end of the 95% interval for the flip rate seen here)")
    if p > det["alpha"]:
        lines.append(f"{flips} is consistent with no real difference (exact McNemar p = {_pval(p)}), though not "
                     f"evidence that there is none{need}.")
    else:
        lines.append(f"{flips} ({b} only {la} right, {c} only {lb} right): the exact McNemar test says the runs "
                     f"differ (p = {_pval(p)}).")
    lo, hi = 100 * s["diff_lo"], 100 * s["diff_hi"]
    def side(x):
        return f"{abs(x):.1f} points {'worse' if x > 0 else 'better'}" if abs(x) >= 0.05 else "equal"
    lines.append(f"Accuracy difference {la} − {lb}: {_pts(s['diff'])} points, 95% interval [{_pts(s['diff_lo'])}, "
                 f"{_pts(s['diff_hi'])}]. The data are compatible with {lb} being anywhere from {side(lo)} to "
                 f"{side(hi)} than {la}" + (f"; drops larger than {hi:.1f} points fall outside the interval, smaller ones do not."
                                            if hi > 0 and lo <= 0 else "."))
    if m < det["min_flips_for_significance"]:
        lines.append(f"With fewer than {det['min_flips_for_significance']} flips no split can reach "
                     f"p ≤ {det['alpha']:g}, even if every flip goes the same way.")
    return lines


def to_markdown(s, la="A", lb="B", match_desc="") -> str:
    la, lb = _cell(la), _cell(lb)
    L = [f"# {la} vs {lb}", ""]
    L.append(f"{s['n']} questions in both runs. Graded by {match_desc}." if match_desc else
             f"{s['n']} questions in both runs.")
    L += [""] + verdict(s, la, lb) + [""]
    L += ["## Accuracy", "", "| run | correct |", "|---|---|",
          f"| {la} | {_ci(s['acc_a'])} |", f"| {lb} | {_ci(s['acc_b'])} |", ""]
    L += ["## Paired table", "",
          f"| | {lb} right | {lb} wrong |", "|---|---|---|",
          f"| **{la} right** | {s['both_right']} (both right) | **{s['b']}** (only {la}: b) |",
          f"| **{la} wrong** | **{s['c']}** (only {lb}: c) | {s['both_wrong']} (both wrong) |", "",
          f"- discordant pairs: b = {s['b']}, c = {s['c']} (flip rate {_pct(s['discordant'] / s['n'])})",
          f"- exact McNemar (binomial, two-sided): p = {_pval(s['mcnemar_exact_p'])}",
          f"- accuracy difference {la} − {lb}: {_pts(s['diff'])} points, 95% CI [{_pts(s['diff_lo'])}, "
          f"{_pts(s['diff_hi'])}] (Tango score interval, paired)", ""]
    a = s.get("answers")
    if a:
        L += ["## Same answers", "", "| measure | value |", "|---|---|",
              f"| word-for-word identical (exact text, ignoring leading/trailing spaces) | {_ci(a['identical'])} |",
              f"| identical after normalising case, punctuation, articles | {_ci(a['identical_normalised'])} |",
              f"| same answer by token overlap (F1 between the two >= 0.5, or both abstain) | {_ci(a['same'])} |",
              f"| **both wrong when the answers are the same (token overlap)** | {_ci(a['wrong_when_same'])} |",
              f"| both wrong when word-for-word identical | {_ci(a['wrong_when_identical'])} |",
              f"| same wrong answer (token overlap), among questions both got wrong | {_ci(a['same_among_both_wrong'])} |",
              "", "Token overlap is a cheap proxy for 'the same answer'; two answers sharing half their words can "
              "still differ in meaning. Word-for-word identity is strict.", ""]
        if a["inconsistent_grades_when_identical"]:
            L += [f"Note: {a['inconsistent_grades_when_identical']} identical answer pair(s) were graded "
                  f"differently; check the 'correct' fields.", ""]
    d = s["detect"]
    L += ["## How many questions to see a small drop", "",
          f"Planning estimates under assumed flip rates, not guarantees: independent questions like these, a true "
          f"drop of the given size, exact McNemar, α = {d['alpha']:g}, power {d['power']:.0%}, {lb} worse than {la} by the given points. "
          f"The flip rate is the share of questions where the two runs disagree on right/wrong: observed "
          f"{_pct(d['flip_rate'])}, upper end of its 95% Wilson interval {_pct(d['flip_rate_hi'])}. A drop of x points needs at least "
          f"x% of questions to flip.", "",
          "| drop | every flip one way (best case) | at observed flip rate | at upper end of flip-rate interval |",
          "|---|---|---|---|"]
    for r in d["plans"]:
        cell = lambda v, ok: _fmt_n(v) if ok else "rate below the drop"
        L.append(f"| {r['points']:g} pt | {_fmt_n(r['best_case'])} | "
                 f"{cell(r['at_observed_rate'], d['flip_rate'] >= r['points'] / 100)} | "
                 f"{cell(r['at_upper_rate'], d['flip_rate_hi'] >= r['points'] / 100)} |")
    L.append("")
    if s.get("by_category"):
        L += ["## By category", "",
              f"| category | n | {la} | {lb} | both right | only {la} | only {lb} | both wrong | McNemar p | {la} − {lb}, points [95% CI] |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for c, r in s["by_category"].items():
            L.append(f"| {_cell(c)} | {r['n']} | {_pct(r['acc_a']['p'])} | {_pct(r['acc_b']['p'])} | {r['both_right']} | "
                     f"{r['only_a']} | {r['only_b']} | {r['both_wrong']} | {_pval(r['mcnemar_exact_p'])} | "
                     f"{_pts(r['diff'])} [{_pts(r['diff_lo'])}, {_pts(r['diff_hi'])}] |")
        L += ["", "Per-category tests are not corrected for multiple comparisons.", ""]
    f = s["flipped"]
    if f["only_a"] or f["only_b"]:
        L += ["## Flipped questions", ""]
        for key, name in (("only_a", la), ("only_b", lb)):
            ids = f[key]
            if ids:
                more = f" (+{len(ids) - 30} more in the JSON output)" if len(ids) > 30 else ""
                L.append(f"- only {name} right: {', '.join('`' + _cell(i).replace('`', '') + '`' for i in ids[:30])}{more}")
        L.append("")
    return "\n".join(L)


def run_compare(path_a, path_b, gold=None, match="exact", threshold=0.5, label_a=None, label_b=None,
                detect=(1.0,), alpha=0.05, power=0.8):
    rows_a, rows_b = load_rows(Path(path_a)), load_rows(Path(path_b))
    gold_rows = load_rows(Path(gold)) if gold else None
    pairs, unmatched = pair_runs(rows_a, rows_b, gold_rows, match, threshold, (str(path_a), str(path_b), str(gold)))
    s = analyze(pairs, detect, alpha, power)
    s["unmatched"] = unmatched
    la, lb = label_a or Path(path_a).stem, label_b or Path(path_b).stem
    s["labels"] = {"a": la, "b": lb}
    if gold:
        desc = {"exact": "exact match to the gold answer after normalisation",
                "contains": "the gold answer appearing in the answer (after normalisation)",
                "f1": f"token F1 against the gold answer >= {threshold:g}"}[match]
    else:
        desc = "the 'correct' field in each run"
    s["graded_by"] = desc
    md = to_markdown(s, la, lb, desc)
    notes = [f"{v} question(s) {k.replace('_', ' ')}" for k, v in unmatched.items() if v]
    if notes:
        md += "\nLeft out: " + "; ".join(notes) + ".\n"
    return s, md
