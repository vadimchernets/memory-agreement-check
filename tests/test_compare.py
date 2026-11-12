import json
import tempfile
import unittest
from pathlib import Path

from memcheck import compare, stats

EX = Path(__file__).resolve().parent.parent / "examples" / "compare"


def _write(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


def _fp8_case(d):
    """120 questions: bf16 right on 72, FP8 right on 71; the one flip is bf16-right / FP8-wrong;
    49 answers word-for-word identical. Synthetic, shaped like a published FP8-vs-bf16 check."""
    a, b = [], []
    for i in range(120):
        a.append({"qid": f"q{i}", "answer": f"alpha {i}", "correct": i < 72})
        b.append({"qid": f"q{i}", "answer": f"alpha {i}" if i < 49 else f"beta gamma {i}", "correct": i < 71})
    pa, pb = Path(d) / "bf16.jsonl", Path(d) / "fp8.jsonl"
    _write(pa, a)
    _write(pb, b)
    return pa, pb


class PairedStats(unittest.TestCase):
    def test_exact_mcnemar(self):
        self.assertEqual(stats.mcnemar_exact(1, 0), 1.0)
        self.assertEqual(stats.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(stats.mcnemar_exact(6, 0), 0.03125)        # 2 * 0.5**6
        self.assertAlmostEqual(stats.mcnemar_exact(5, 0), 0.0625)
        self.assertAlmostEqual(stats.mcnemar_exact(10, 2), 0.03857421875)  # 2 * P(Bin(12, .5) <= 2)
        self.assertEqual(stats.mcnemar_exact(3, 7), stats.mcnemar_exact(7, 3))

    def test_tango_interval_matches_reference(self):
        # Bentur et al. (2009) example in Fagerland, Lydersen & Laake: n12 = 1, n21 = 7, N = 21;
        # R contingencytables: Tango asymptotic score CI -0.2857 (-0.5172 to -0.0260)
        d, lo, hi = stats.paired_diff_ci(1, 1, 7, 12)
        self.assertAlmostEqual(d, -6 / 21)
        self.assertAlmostEqual(lo, -0.5172, places=4)
        self.assertAlmostEqual(hi, -0.0260, places=4)
        # no flips at all: still a real interval, symmetric around 0 (z^2 / (n + z^2) = 0.161 for n = 20)
        d, lo, hi = stats.paired_diff_ci(10, 0, 0, 10)
        self.assertEqual(d, 0.0)
        self.assertAlmostEqual(hi, 0.1611, places=4)
        self.assertAlmostEqual(lo, -hi, places=6)
        d, lo, hi = stats.paired_diff_ci(0, 5, 0, 0)
        self.assertEqual(hi, 1.0)

    def test_power_and_sample_size(self):
        # every flip one way, 1-point drop: b ~ Bin(n, 0.01), c = 0; exact test needs b >= 6
        n = stats.n_for_drop(0.01, 0.01)
        self.assertTrue(700 <= n <= 900, n)
        self.assertGreaterEqual(stats.mcnemar_power(n, 0.01, 0.0), 0.8)
        self.assertLess(stats.mcnemar_power(n // 2, 0.01, 0.0), 0.8)
        # more flips in both directions make a 1-point drop harder to see; close to the asymptotic formula
        n5 = stats.n_for_drop(0.01, 0.05)
        self.assertTrue(3800 <= n5 <= 4400, n5)
        p10, p01 = 0.03, 0.02                                        # power crosses the target at the returned n
        self.assertGreaterEqual(stats.mcnemar_power(n5, p10, p01), 0.8)
        self.assertLess(stats.mcnemar_power(n5 - 1, p10, p01), 0.8)
        self.assertAlmostEqual(stats.binom_cdf(3, 10, 0.5), 176 / 1024, places=12)
        self.assertAlmostEqual(stats.binom_sf(7, 10, 0.3), sum(stats.binom_pmf(i, 10, 0.3) for i in range(7, 11)))
        self.assertIsNone(stats.n_for_drop(0.02, 0.01))   # a 2-point drop needs at least 2% of answers to flip
        # rejection threshold: largest k with 2 * P(X <= k) <= alpha, exact and normal branches agree
        self.assertEqual(stats._mcnemar_reject_k(438, 0.05), 198)   # where a normal approximation is off by one
        for m in (20, 401, 438, 1000):
            k = stats._mcnemar_reject_k(m, 0.05)
            self.assertLessEqual(2 * stats.binom_cdf(k, m, 0.5), 0.05)
            self.assertGreater(2 * stats.binom_cdf(k + 1, m, 0.5), 0.05)


class CompareRuns(unittest.TestCase):
    def test_one_flip_in_120(self):
        with tempfile.TemporaryDirectory() as d:
            pa, pb = _fp8_case(d)
            s, md = compare.run_compare(pa, pb, label_a="bf16", label_b="FP8")
        self.assertEqual((s["both_right"], s["b"], s["c"], s["both_wrong"]), (71, 1, 0, 48))
        self.assertEqual(s["mcnemar_exact_p"], 1.0)
        self.assertAlmostEqual(100 * s["diff"], 0.833, places=2)
        self.assertLess(s["diff_lo"], 0)                       # includes zero
        self.assertGreater(100 * s["diff_hi"], 2.0)            # and a drop of over 2 points: wide
        self.assertEqual(s["answers"]["identical"]["k"], 49)
        self.assertEqual(s["detect"]["min_flips_for_significance"], 6)
        n1 = s["detect"]["plans"][0]["best_case"]
        self.assertIn("1 flip out of 120 is consistent with no real difference", md)
        self.assertIn(f"you need N ≈ {n1:,} questions to detect a 1-point drop", md)
        self.assertIn("With fewer than 6 flips no split can reach", md)
        self.assertIn("| **bf16 right** | 71 (both right) | **1** (only bf16: b) |", md)

    def test_grading_with_gold_categories_and_agreement(self):
        with tempfile.TemporaryDirectory() as d:
            _write(Path(d) / "g.jsonl", [
                {"qid": "1", "answer": "Paris", "category": "geo"},
                {"qid": "2", "answer": ["4", "four"], "category": "math"},
                {"qid": "3", "answer": "blue", "category": "geo"},
                {"qid": "4", "answer": "7", "category": "math"},
                {"qid": "5", "answer": "x"}])                      # not answered by B: left out
            _write(Path(d) / "a.jsonl", [{"qid": 1, "answer": "paris."}, {"qid": 2, "answer": "Four"},
                                         {"qid": 3, "answer": "red"}, {"qid": 4, "answer": "8"},
                                         {"qid": 5, "answer": "x"}])
            _write(Path(d) / "b.jsonl", [{"id": "1", "prediction": "Lyon"}, {"id": "2", "prediction": "4"},
                                         {"id": "3", "prediction": "red"}, {"id": "4", "prediction": "7"}])
            s, md = compare.run_compare(Path(d) / "a.jsonl", Path(d) / "b.jsonl", Path(d) / "g.jsonl", detect=())
        self.assertEqual(s["n"], 4)
        self.assertEqual(s["unmatched"]["only_in_a"], 1)
        self.assertEqual((s["both_right"], s["b"], s["c"], s["both_wrong"]), (1, 1, 1, 1))
        a = s["answers"]
        self.assertEqual(a["identical"]["k"], 1)                  # "red" == "red"
        self.assertEqual(a["wrong_when_identical"]["k"], 1)       # ...and both wrong
        self.assertEqual(a["same_among_both_wrong"]["k"], 1)
        self.assertEqual(set(s["by_category"]), {"geo", "math"})
        self.assertEqual(s["by_category"]["geo"]["b"], 1)
        self.assertEqual(s["by_category"]["math"]["c"], 1)
        self.assertIn("## By category", md)

    def test_match_modes(self):
        self.assertTrue(compare.grade("The capital is Paris.", "paris", "contains"))
        self.assertFalse(compare.grade("Parisian", "paris", "contains"))
        self.assertFalse(compare.grade("The capital is Paris.", "paris", "exact"))
        self.assertTrue(compare.grade("the Eiffel tower", "Eiffel tower", "f1"))

    def test_errors(self):
        with tempfile.TemporaryDirectory() as d:
            _write(Path(d) / "a.jsonl", [{"qid": "1", "answer": "x"}])
            _write(Path(d) / "b.jsonl", [{"qid": "1", "answer": "x", "correct": True}])
            with self.assertRaises(ValueError):          # no gold and no 'correct' in A
                compare.run_compare(Path(d) / "a.jsonl", Path(d) / "b.jsonl")
            _write(Path(d) / "c.jsonl", [{"qid": "1", "correct": 1}, {"qid": "1", "correct": 0}])
            with self.assertRaises(ValueError):          # duplicate qid
                compare.run_compare(Path(d) / "c.jsonl", Path(d) / "b.jsonl")
            _write(Path(d) / "g.jsonl", [{"qid": "1", "answer": "x", "category": "geo"}])
            _write(Path(d) / "e.jsonl", [{"qid": "1", "correct": True}])
            with self.assertRaises(ValueError):          # --gold but no answer: not silently 'wrong'
                compare.run_compare(Path(d) / "e.jsonl", Path(d) / "a.jsonl", Path(d) / "g.jsonl")
            _write(Path(d) / "f.jsonl", [{"qid": "1", "answer": "x", "category": "math"}])
            with self.assertRaises(ValueError):          # categories disagree between files
                compare.run_compare(Path(d) / "f.jsonl", Path(d) / "a.jsonl", Path(d) / "g.jsonl")

    def test_markdown_is_escaped(self):
        with tempfile.TemporaryDirectory() as d:
            _write(Path(d) / "a.jsonl", [{"qid": "x|y", "correct": True, "category": "a|b"},
                                         {"qid": "z", "correct": False}])
            _write(Path(d) / "b.jsonl", [{"qid": "x|y", "correct": False}, {"qid": "z", "correct": False}])
            s, md = compare.run_compare(Path(d) / "a.jsonl", Path(d) / "b.jsonl", label_a="A|1", detect=())
        self.assertIn("| a\\|b |", md)
        self.assertIn("`x\\|y`", md)
        self.assertIn("(no category)", s["by_category"])
        self.assertNotIn("| A|1 |", md)

    def test_cli_on_example(self):
        from memcheck.cli import main
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "r.json"
            main(["compare", str(EX / "run_bf16.jsonl"), str(EX / "run_fp8.jsonl"), "--gold", str(EX / "gold.jsonl"),
                  "--match", "contains", "--json", str(out), "--detect", "1"])
            s = json.loads(out.read_text())
        self.assertEqual(s["n"], 150)
        self.assertEqual(s["b"] + s["c"] + s["both_right"] + s["both_wrong"], 150)
        self.assertIn("by_category", s)


if __name__ == "__main__":
    unittest.main()
