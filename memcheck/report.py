"""Turn records.jsonl into summary.json and summary.md."""
from __future__ import annotations

import json
from pathlib import Path

from .locomo import CATEGORIES
from .stats import bootstrap, phi, wilson

INCOMPLETE = ("partial", "none")


def load_records(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _wrong_a(r):
    return not r["final"]["a_correct"]


def _wrong_b(r):
    return not r["final"]["b_correct"]


def _agree(r):
    return r["final"]["agree"]


def _agree_wrong(r):
    return _agree(r) and _wrong_a(r)   # agreeing pairs have equal correctness by construction


def _substantive(r):
    return not (r["a"]["abstain"] and r["b"]["abstain"])


def _p_wrong_given_agree(rs):
    ag = [r for r in rs if _agree(r)]
    return (sum(map(_agree_wrong, ag)) / len(ag)) if ag else None


def _p_wrong_given_subst_agree(rs):
    ag = [r for r in rs if _agree(r) and _substantive(r)]
    return (sum(map(_agree_wrong, ag)) / len(ag)) if ag else None


def _phi(rs):
    return phi([_wrong_a(r) for r in rs], [_wrong_b(r) for r in rs])


def group(rs: list[dict], seed: int = 0) -> dict:
    n = len(rs)
    if n == 0:
        return {"n": 0}
    ka, kb = sum(not _wrong_a(r) for r in rs), sum(not _wrong_b(r) for r in rs)
    ag = [r for r in rs if _agree(r)]
    agw = [r for r in ag if _agree_wrong(r)]
    sag = [r for r in ag if _substantive(r)]
    sagw = [r for r in sag if _agree_wrong(r)]
    both_wrong = sum(1 for r in rs if _wrong_a(r) and _wrong_b(r))
    pa, pb = ka / n, kb / n
    ph = _phi(rs)
    ph_ci = bootstrap(rs, _phi, seed=seed, strata=lambda r: r["category"])
    return {
        "n": n,
        "acc_a": wilson(ka, n), "acc_b": wilson(kb, n),
        "score_a": sum(r["rule"]["a_score"] for r in rs) / n, "score_b": sum(r["rule"]["b_score"] for r in rs) / n,
        "agree": wilson(len(ag), n),
        "wrong_given_agree": wilson(len(agw), len(ag)),
        "agree_and_wrong": wilson(len(agw), n),
        "n_agree": len(ag), "n_agree_wrong": len(agw),
        "wrong_given_substantive_agree": wilson(len(sagw), len(sag)),
        "n_substantive_agree": len(sag), "n_substantive_agree_wrong": len(sagw),
        "both_wrong": wilson(both_wrong, n),
        "both_wrong_if_independent": (1 - pa) * (1 - pb),
        "error_phi": ph, "error_phi_ci": ph_ci,
        "n_eff_two_models": (2 / (1 + ph)) if ph is not None and ph > -1 else None,
        "judge_inconsistent": sum(r["final"]["judge_inconsistent"] for r in rs),
        "tool_used": sum(r["a"]["tool_used"] or r["b"]["tool_used"] for r in rs),
    }


def summarize(records: list[dict], seed: int = 0) -> dict:
    rs = records
    non_adv = [r for r in rs if r["category"] != 5]
    full = [r for r in non_adv if r["retrieval"]["status"] == "full"]
    inc = [r for r in non_adv if r["retrieval"]["status"] in INCOMPLETE]

    def diff(stat):
        def f(s):
            a = stat([r for r in s if r["retrieval"]["status"] in INCOMPLETE])
            b = stat([r for r in s if r["retrieval"]["status"] == "full"])
            return None if a is None or b is None else a - b
        return f(non_adv), bootstrap(non_adv, f, seed=seed, strata=lambda r: r["category"])

    d_wga = diff(_p_wrong_given_agree)
    d_wgsa = diff(_p_wrong_given_subst_agree)
    agw = [r for r in non_adv if _agree_wrong(r)]
    agw_sub = [r for r in agw if _substantive(r)]
    return {
        "n_records": len(rs),
        "judge_source": {k: sum(r["final"]["source"] == k for r in rs) for k in ("judge", "rule")},
        "retrieval_status": {k: sum(r["retrieval"]["status"] == k for r in rs)
                             for k in ("full", "partial", "none", "unknown")},
        "all": group(rs, seed),
        "non_adversarial": group(non_adv, seed),
        "by_category": {CATEGORIES[c]: group([r for r in rs if r["category"] == c], seed)
                        for c in sorted({r["category"] for r in rs})},
        "retrieval_full_non_adv": group(full, seed),
        "retrieval_incomplete_non_adv": group(inc, seed),
        "diff_wrong_given_agree_incomplete_minus_full": {"point": d_wga[0], "ci": d_wga[1]},
        "diff_wrong_given_substantive_agree_incomplete_minus_full": {"point": d_wgsa[0], "ci": d_wgsa[1]},
        "agreed_wrong_non_adv_with_incomplete_retrieval":
            wilson(sum(r["retrieval"]["status"] in INCOMPLETE for r in agw), len(agw)),
        "substantive_agreed_wrong_non_adv_with_incomplete_retrieval":
            wilson(sum(r["retrieval"]["status"] in INCOMPLETE for r in agw_sub), len(agw_sub)),
    }


def _pct(t):
    if not t or t[0] is None:
        return "–"
    p, lo, hi = t
    return f"{100 * p:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def _row(name, g):
    if not g.get("n"):
        return f"| {name} | 0 |" + " – |" * 8
    ph = g["error_phi"]
    ci = g["error_phi_ci"]
    phs = "–" if ph is None else (f"{ph:.2f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci and ci[0] is not None else ""))
    return (f"| {name} | {g['n']} | {_pct(g['acc_a'])} | {_pct(g['acc_b'])} | {_pct(g['agree'])} | "
            f"{_pct(g['wrong_given_agree'])} ({g['n_agree_wrong']}/{g['n_agree']}) | "
            f"{_pct(g['wrong_given_substantive_agree'])} ({g['n_substantive_agree_wrong']}/{g['n_substantive_agree']}) | "
            f"{_pct(g['both_wrong'])} vs {100 * g['both_wrong_if_independent']:.1f}% | {phs} |")


HEAD = ("| group | n | model A correct | model B correct | agree | wrong when they agree | "
        "wrong when they agree on a real answer | both wrong (observed vs if independent) | error correlation φ |\n"
        "|---|---|---|---|---|---|---|---|---|")


def markdown(s: dict, meta: dict) -> str:
    L = [f"# Summary: {meta.get('run_name', '')}", "",
         f"- Model A: `{meta['model_a']}`; model B: `{meta['model_b']}`; judge: `{meta['judge']}`",
         f"- Memory: `{meta.get('memory')}`; questions: {s['n_records']} (seed {meta['seed']}, "
         f"{'equal number per category' if meta.get('balanced') else 'random'}); LoCoMo commit `{meta['locomo_commit'][:10]}`",
         f"- Correctness from: {s['judge_source']}; stemmer: {meta.get('stemmer')}",
         f"- Retrieval of LoCoMo evidence turns: {s['retrieval_status']}",
         "- Intervals: Wilson 95% for proportions; stratified bootstrap 95% (2,000 resamples) for φ and differences.",
         "- \"Agree\" = the two answers say the same thing and get the same grade. \"Real answer\" = not both "
         "\"Not mentioned\".", "", "## Overall and by category", "", HEAD,
         _row("all questions", s["all"]), _row("all except adversarial", s["non_adversarial"])]
    for name, g in s["by_category"].items():
        L.append(_row(name, g))
    L += ["", "## By retrieval quality (categories 1-4)", "",
          "\"Full\": every evidence turn LoCoMo marks for the question was in the retrieved memory. "
          "\"Incomplete\": at least one was missing.", "", HEAD,
          _row("evidence fully retrieved", s["retrieval_full_non_adv"]),
          _row("evidence incomplete", s["retrieval_incomplete_non_adv"]), ""]
    d, ds = s["diff_wrong_given_agree_incomplete_minus_full"], s["diff_wrong_given_substantive_agree_incomplete_minus_full"]

    def pp(x):
        if x["point"] is None:
            return "–"
        ci = x["ci"]
        return f"{100 * x['point']:+.1f} points" + (f" [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]" if ci and ci[0] is not None else "")

    L += [f"- Wrong-when-agreeing, incomplete minus full retrieval: {pp(d)}",
          f"- Same, counting only agreement on a real answer: {pp(ds)}",
          f"- Share of agreed-but-wrong answers (categories 1-4) that came with incomplete retrieval: "
          f"{_pct(s['agreed_wrong_non_adv_with_incomplete_retrieval'])}",
          f"- Same, agreement on a real answer only: {_pct(s['substantive_agreed_wrong_non_adv_with_incomplete_retrieval'])}",
          ""]
    return "\n".join(L)


def write(out: Path, seed: int = 0) -> dict:
    out = Path(out)
    recs = load_records(out / "records.jsonl")
    meta = json.loads((out / "meta.json").read_text())
    s = summarize(recs, seed)
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    (out / "summary.md").write_text(markdown(s, meta))
    return s
