"""Turn records.jsonl into summary.json and summary.md.

Everything is recomputed from the stored answer texts and grades, so improvements to scoring
(e.g. abstention detection) apply to old runs without calling any model again."""
from __future__ import annotations

import json
from pathlib import Path

from . import scoring
from .locomo import CATEGORIES
from .stats import bootstrap, phi, wilson

DEFAULT_SEED = 2026
INCOMPLETE = ("partial", "none")
MIN_N = 5   # percentage cells with fewer cases are shown as counts only


def load_records(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _gold_by_qid(data_path) -> dict:
    try:
        from .locomo import default_data_path, load
        _, qs = load(Path(data_path) if data_path else default_data_path())
        return {q.qid: q.answer for q in qs}
    except Exception:  # noqa: BLE001 - dataset absent: fall back to the stored rule grades
        return {}


def rows(records: list[dict], view: str, gold: dict) -> list[dict]:
    """view = 'judge' (judge grades and 'same'; rule where the judge failed) or 'rule' (LoCoMo F1 >= 0.5)."""
    out = []
    for r in records:
        ta, tb, cat = r["a"]["text"], r["b"]["text"], r["category"]
        g = gold.get(r["qid"])
        if g is not None:
            rc_a, rc_b = scoring.rule_correct(ta, g, cat), scoring.rule_correct(tb, g, cat)
            sa, sb = scoring.locomo_score(ta, g, cat), scoring.locomo_score(tb, g, cat)
        else:
            rc_a, rc_b = r["rule"]["a_correct"], r["rule"]["b_correct"]
            sa, sb = r["rule"]["a_score"], r["rule"]["b_score"]
        j = r.get("judge") or {}
        if view == "judge" and j and not j.get("failed"):
            ca, cb, same, src = j["a_correct"], j["b_correct"], j["same"], "judge"
        else:
            ca, cb, same, src = rc_a, rc_b, scoring.rule_same(ta, tb), "rule"
        out.append({"qid": r["qid"], "category": cat, "status": r["retrieval"]["status"],
                    "a_ok": bool(ca), "b_ok": bool(cb), "agree": bool(same), "src": src,
                    "abstain_both": scoring.is_abstain(ta) and scoring.is_abstain(tb),
                    "score_a": sa, "score_b": sb, "inconsistent": bool(same and ca != cb),
                    "tool_used": r["a"]["tool_used"] or r["b"]["tool_used"]})
    return out


def _aw(x):  # agreed and both wrong
    return x["agree"] and not x["a_ok"] and not x["b_ok"]


def _p_wrong_given_agree(rs, substantive=False):
    ag = [x for x in rs if x["agree"] and not (substantive and x["abstain_both"])]
    return (sum(map(_aw, ag)) / len(ag)) if ag else None


def _phi(rs):
    return phi([not x["a_ok"] for x in rs], [not x["b_ok"] for x in rs])


def group(rs: list[dict], seed: int = DEFAULT_SEED) -> dict:
    n = len(rs)
    if n == 0:
        return {"n": 0}
    ka, kb = sum(x["a_ok"] for x in rs), sum(x["b_ok"] for x in rs)
    ag = [x for x in rs if x["agree"]]
    agw = [x for x in ag if _aw(x)]
    sag = [x for x in ag if not x["abstain_both"]]
    sagw = [x for x in sag if _aw(x)]
    both_wrong = sum(1 for x in rs if not x["a_ok"] and not x["b_ok"])
    pa, pb = ka / n, kb / n
    ph = _phi(rs)
    return {
        "n": n, "acc_a": wilson(ka, n), "acc_b": wilson(kb, n),
        "f1_a": sum(x["score_a"] for x in rs) / n, "f1_b": sum(x["score_b"] for x in rs) / n,
        "agree": wilson(len(ag), n), "n_agree": len(ag), "n_agree_wrong": len(agw),
        "wrong_given_agree": wilson(len(agw), len(ag)),
        "n_substantive_agree": len(sag), "n_substantive_agree_wrong": len(sagw),
        "wrong_given_substantive_agree": wilson(len(sagw), len(sag)),
        "both_wrong": wilson(both_wrong, n), "both_wrong_if_independent": (1 - pa) * (1 - pb),
        "error_phi": ph, "error_phi_ci": bootstrap(rs, _phi, seed=seed, strata=lambda x: x["category"]),
        "n_eff_heuristic": (2 / (1 + ph)) if ph is not None and ph > -1 else None,
        "judge_inconsistent": sum(x["inconsistent"] for x in rs),
        "tool_used": sum(x["tool_used"] for x in rs),
    }


def _split_diff(rs, substantive, seed):
    def f(s):
        a = _p_wrong_given_agree([x for x in s if x["status"] in INCOMPLETE], substantive)
        b = _p_wrong_given_agree([x for x in s if x["status"] == "full"], substantive)
        return None if a is None or b is None else a - b

    def std(s):  # equal-weight average of the within-category differences (categories where both exist)
        ds = [f([x for x in s if x["category"] == c]) for c in sorted({x["category"] for x in s})]
        ds = [d for d in ds if d is not None]
        return sum(ds) / len(ds) if ds else None

    strata = lambda x: (x["category"], x["status"] == "full")  # noqa: E731
    return {"pooled": f(rs), "pooled_ci": bootstrap(rs, f, seed=seed, strata=strata),
            "category_standardized": std(rs), "category_standardized_ci": bootstrap(rs, std, seed=seed, strata=strata)}


def view_summary(rs: list[dict], seed: int) -> dict:
    non_adv = [x for x in rs if x["category"] != 5]
    full = [x for x in non_adv if x["status"] == "full"]
    inc = [x for x in non_adv if x["status"] in INCOMPLETE]
    agw = [x for x in non_adv if _aw(x)]
    agw_s = [x for x in agw if not x["abstain_both"]]
    return {
        "grading": {k: sum(x["src"] == k for x in rs) for k in ("judge", "rule")},
        "all": group(rs, seed), "non_adversarial": group(non_adv, seed),
        "by_category": {CATEGORIES[c]: group([x for x in rs if x["category"] == c], seed)
                        for c in sorted({x["category"] for x in rs})},
        "retrieval_full_non_adv": group(full, seed), "retrieval_incomplete_non_adv": group(inc, seed),
        "by_category_and_retrieval": {
            CATEGORIES[c]: {"full": group([x for x in full if x["category"] == c], seed),
                            "incomplete": group([x for x in inc if x["category"] == c], seed)}
            for c in sorted({x["category"] for x in non_adv})},
        "diff_wrong_given_agree": _split_diff(non_adv, False, seed),
        "diff_wrong_given_substantive_agree": _split_diff(non_adv, True, seed),
        "agreed_wrong_with_incomplete_retrieval": wilson(sum(x["status"] in INCOMPLETE for x in agw), len(agw)),
        "substantive_agreed_wrong_with_incomplete_retrieval":
            wilson(sum(x["status"] in INCOMPLETE for x in agw_s), len(agw_s)),
        "n_agreed_wrong_both_abstain": sum(x["abstain_both"] for x in agw), "n_agreed_wrong": len(agw),
    }


def summarize(records: list[dict], seed: int = DEFAULT_SEED, data_path=None) -> dict:
    gold = _gold_by_qid(data_path)
    return {
        "n_records": len(records),
        "rule_recomputed_from_dataset": bool(gold),
        "retrieval_status": {k: sum(r["retrieval"]["status"] == k for r in records)
                             for k in ("full", "partial", "none", "unknown")},
        "retrieval_unknown_qids": [r["qid"] for r in records if r["retrieval"]["status"] == "unknown"],
        "judge": view_summary(rows(records, "judge", gold), seed),
        "rule": view_summary(rows(records, "rule", gold), seed),
    }


def _pct(t, k=None, n=None):
    if not t or t[0] is None:
        return "–"
    if n is not None and n < MIN_N:
        return f"{k}/{n}"
    p, lo, hi = t
    s = f"{100 * p:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"
    return s + (f" ({k}/{n})" if n is not None else "")


def _phis(g):
    ph, ci = g["error_phi"], g["error_phi_ci"]
    if ph is None:
        return "–"
    return f"{ph:.2f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci and ci[0] is not None else "")


def _row(name, g):
    if not g.get("n"):
        return f"| {name} | 0 |" + " – |" * 7
    return (f"| {name} | {g['n']} | {_pct(g['acc_a'])} | {_pct(g['acc_b'])} | {_pct(g['agree'])} | "
            f"{_pct(g['wrong_given_substantive_agree'], g['n_substantive_agree_wrong'], g['n_substantive_agree'])} | "
            f"{_pct(g['wrong_given_agree'], g['n_agree_wrong'], g['n_agree'])} | "
            f"{_pct(g['both_wrong'])} vs {100 * g['both_wrong_if_independent']:.1f}% | {_phis(g)} |")


HEAD = ("| group | n | model A correct | model B correct | agree | wrong when they agree on a real answer | "
        "wrong when they agree (incl. both \"Not mentioned\") | both wrong (observed vs if independent) | "
        "error correlation φ |\n|---|---|---|---|---|---|---|---|---|")


def _pp(v, ci):
    if v is None:
        return "–"
    return f"{100 * v:+.1f} points" + (f" [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]" if ci and ci[0] is not None else "")


def _cell(g, substantive):
    if not g.get("n"):
        return "–"
    k, n = (g["n_substantive_agree_wrong"], g["n_substantive_agree"]) if substantive else (g["n_agree_wrong"], g["n_agree"])
    return f"{k}/{n}" + (f" ({100 * k / n:.0f}%)" if n >= MIN_N else "")


def _view_md(v: dict, title: str) -> list[str]:
    L = [f"## {title}", "", f"Grades from: {v['grading']}", "", "### Overall and by category", "", HEAD,
         _row("all questions", v["all"]), _row("all except adversarial", v["non_adversarial"])]
    L += [_row(name, g) for name, g in v["by_category"].items()]
    L += ["", "### By retrieval quality (categories 1-4)", "", HEAD,
          _row("evidence fully retrieved", v["retrieval_full_non_adv"]),
          _row("evidence incomplete", v["retrieval_incomplete_non_adv"]), "",
          "Within each category, agreed-and-wrong / agreed:", "",
          "| category | full memory: real answer | full: incl. \"Not mentioned\" | incomplete: real answer | "
          "incomplete: incl. \"Not mentioned\" |", "|---|---|---|---|---|"]
    for c, d in v["by_category_and_retrieval"].items():
        L.append(f"| {c} | {_cell(d['full'], True)} | {_cell(d['full'], False)} | "
                 f"{_cell(d['incomplete'], True)} | {_cell(d['incomplete'], False)} |")
    ds, da = v["diff_wrong_given_substantive_agree"], v["diff_wrong_given_agree"]
    L += ["",
          f"- Wrong-when-agreeing on a real answer, incomplete minus full retrieval: pooled "
          f"{_pp(ds['pooled'], ds['pooled_ci'])}; category-standardized "
          f"{_pp(ds['category_standardized'], ds['category_standardized_ci'])}",
          f"- Same including both-\"Not mentioned\" pairs: pooled {_pp(da['pooled'], da['pooled_ci'])}; "
          f"category-standardized {_pp(da['category_standardized'], da['category_standardized_ci'])}",
          f"- Agreed-but-wrong pairs (categories 1-4): {v['n_agreed_wrong']}, of which both \"Not mentioned\": "
          f"{v['n_agreed_wrong_both_abstain']}",
          f"- Share of agreed-but-wrong pairs with incomplete retrieval: "
          f"{_pct(v['agreed_wrong_with_incomplete_retrieval'])}; real answers only: "
          f"{_pct(v['substantive_agreed_wrong_with_incomplete_retrieval'])}", ""]
    return L


def markdown(s: dict, meta: dict) -> str:
    L = [f"# Summary: {meta.get('run_name', '')}", "",
         f"- Model A: `{meta['model_a']}`; model B: `{meta['model_b']}`; judge: `{meta['judge']}`",
         f"- Resolved models: {meta.get('resolved_models', '–')}; CLI versions: {meta.get('cli_versions', '–')}",
         f"- Memory: `{meta.get('memory')}`; questions: {s['n_records']} (seed {meta['seed']}, "
         f"{'equal number per category' if meta.get('balanced') else 'random'}); LoCoMo commit `{meta['locomo_commit'][:10]}`",
         f"- Retrieval of LoCoMo evidence turns: {s['retrieval_status']}. Questions with no usable evidence ids "
         f"(left out of the retrieval split): {', '.join(s['retrieval_unknown_qids']) or 'none'}",
         "- \"Agree\" = the judge (or, in the rule view, token F1 between the two answers >= 0.5) says the two "
         "answers say the same thing. \"Wrong when they agree\" = they agree and both are graded wrong. "
         "\"Real answer\" = not both \"Not mentioned\".",
         "- Intervals: Wilson 95% for proportions (conditional ones treat the number of agreeing pairs as fixed); "
         "stratified bootstrap 95% (2,000 resamples) for φ and differences. Cells with n < 5 show counts only.",
         "- φ is the correlation of the two models' error indicators. 2/(1+φ) is a heuristic for how many "
         "independent checks the pair is worth (the design-effect formula), not a measured quantity.", ""]
    L += _view_md(s["judge"], "View 1: third-company LLM judge")
    L += _view_md(s["rule"], "View 2: LoCoMo token-F1 rule (correct = F1 >= 0.5; adversarial = says \"not mentioned\")")
    g = s["rule"]["all"]
    L += [f"Mean LoCoMo F1 (the official metric) over all questions: model A {g['f1_a']:.3f}, "
          f"model B {g['f1_b']:.3f}.", ""]
    return "\n".join(L)


def write(out: Path, seed: int = DEFAULT_SEED, data_path=None) -> dict:
    out = Path(out)
    recs = load_records(out / "records.jsonl")
    meta = json.loads((out / "meta.json").read_text())
    models: dict = {}
    for r in recs:
        for side in ("a", "b"):
            models.setdefault(side, set()).add(r[side]["model"])
        if r.get("judge"):
            models.setdefault("judge", set()).add(r["judge"].get("model"))
    meta["resolved_models"] = {k: sorted(map(str, v)) for k, v in models.items()}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    s = summarize(recs, seed, data_path)
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    (out / "summary.md").write_text(markdown(s, meta))
    return s
