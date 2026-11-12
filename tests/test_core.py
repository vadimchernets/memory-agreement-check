import json
import tempfile
import unittest
from pathlib import Path

from memcheck import locomo, report, scoring, stats
from memcheck.backends import CliBackend, DryBackend
from memcheck.retrieval import BM25, ExternalMemory, Retrieved, SimpleMemory, coverage
from memcheck.run import parse_judge

FIX = Path(__file__).parent / "fixture_locomo_mini.json"


class Scoring(unittest.TestCase):
    def test_normalize_and_f1(self):
        self.assertEqual(scoring.normalize_answer("The Cat, and a Dog!"), "cat dog")
        self.assertEqual(scoring.f1_score("Pixel", "pixel."), 1.0)
        self.assertAlmostEqual(scoring.f1_score("grey cat Pixel", "Pixel"), 0.5)

    def test_multi_hop_and_open_domain(self):
        self.assertAlmostEqual(scoring.locomo_score("Lisbon, Porto", "Lisbon, Porto", 1), 1.0)
        self.assertAlmostEqual(scoring.locomo_score("Lisbon", "Lisbon, Porto", 1), 0.5)
        self.assertEqual(scoring.locomo_score("yes", "yes; because she likes it", 3), 1.0)

    def test_adversarial_and_abstain(self):
        self.assertEqual(scoring.locomo_score("Not mentioned", "x", 5), 1.0)
        self.assertEqual(scoring.locomo_score("piano", "x", 5), 0.0)
        self.assertFalse(scoring.rule_correct("Not mentioned", "piano", 4))
        self.assertTrue(scoring.rule_same("Not mentioned.", "not mentioned"))
        self.assertFalse(scoring.rule_same("Not mentioned", "piano"))
        self.assertTrue(scoring.rule_same("the piano", "Piano"))
        self.assertFalse(scoring.is_abstain("When Jolene was 10 (the exact year is not mentioned)"))
        self.assertTrue(scoring.is_abstain("Not mentioned. The excerpts only talk about John."))


class Data(unittest.TestCase):
    def setUp(self):
        self.convs, self.qs = locomo.load(FIX)

    def test_load(self):
        self.assertEqual(len(self.convs["mini-1"]), 6)
        self.assertEqual(self.qs[4].answer, locomo.ABSTAIN_GOLD)
        self.assertEqual(self.qs[5].evidence, ["D2:2", "D1:1"])
        self.assertIn("[shared an image: a man holding a medal]", self.convs["mini-1"][3].render())

    def test_sample_is_seeded_and_balanced(self):
        a = locomo.sample(self.qs, 5, seed=1)
        self.assertEqual(sorted(q.category for q in a), [1, 2, 3, 4, 5])
        self.assertEqual([q.qid for q in a], [q.qid for q in locomo.sample(self.qs, 5, seed=1)])
        self.assertEqual(len(locomo.sample(self.qs, 0, seed=1)), 6)


class Retrieval(unittest.TestCase):
    def setUp(self):
        self.convs, self.qs = locomo.load(FIX)
        self.tb = {t.dia_id: t for t in self.convs["mini-1"]}

    def test_bm25_ranks_relevant_first(self):
        bm = BM25(["the cat sleeps", "the marathon in Lisbon", "piano lessons"])
        self.assertEqual(bm.top("Lisbon marathon", 1), [1])

    def test_simple_memory_and_coverage(self):
        mem = SimpleMemory(self.convs, k=1)
        r = mem.retrieve(self.qs[0])
        self.assertEqual(r.dia_ids, ["D1:1"])
        self.assertEqual(coverage(self.qs[0], r, self.tb)["status"], "full")
        self.assertEqual(coverage(self.qs[2], r, self.tb)["status"], "none")
        r2 = Retrieved(context="x", dia_ids=["D1:2"])
        self.assertEqual(coverage(self.qs[2], r2, self.tb)["status"], "partial")

    def test_external_text_match(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ctx.jsonl"
            p.write_text(json.dumps({"qid": "mini-1/0", "memories": [
                {"text": "Ana: I adopted a grey cat called Pixel yesterday.", "score": 0.9}]}) + "\n")
            mem = ExternalMemory(p)
            r = mem.retrieve(self.qs[0])
            cov = coverage(self.qs[0], r, self.tb)
            self.assertEqual((cov["status"], cov["method"]), ("full", "text"))


class Parsing(unittest.TestCase):
    def test_parse_judge(self):
        t = 'thinking...\n• {"answer_1_correct": true, "answer_2_correct": false, "same": false}\nmore'
        self.assertEqual(parse_judge(t), {"answer_1_correct": True, "answer_2_correct": False, "same": False})
        self.assertIsNone(parse_judge("no json here"))
        # string booleans are read strictly: bool("false") would be True
        self.assertEqual(parse_judge('{"answer_1_correct": "false", "answer_2_correct": "true", "same": "false"}'),
                         {"answer_1_correct": False, "answer_2_correct": True, "same": False})
        self.assertIsNone(parse_judge('{"answer_1_correct": "maybe", "answer_2_correct": true, "same": true}'))

    def test_cli_parsers(self):
        cx = CliBackend("codex", None, ".")
        out = "\n".join([json.dumps({"type": "item.completed", "item": {"type": "command_execution"}}),
                         json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "Pixel"}})])
        self.assertEqual(cx._parse(out)[0], "Pixel")
        self.assertTrue(cx._parse(out)[2])
        km = CliBackend("kimi", None, ".")
        kout = "\n".join([json.dumps({"role": "meta", "type": "system.version"}),
                          json.dumps({"role": "assistant", "content": [{"type": "think", "think": "hm"},
                                                                       {"type": "text", "text": "Pixel"}]})])
        self.assertEqual(km._parse(kout)[0], "Pixel")
        cl = CliBackend("claude", "sonnet", ".")
        self.assertEqual(cl._parse(json.dumps({"result": "Pixel", "modelUsage": {"m": {}}}))[0], "Pixel")


class Stats(unittest.TestCase):
    def test_wilson_and_phi(self):
        p, lo, hi = stats.wilson(10, 20)
        self.assertAlmostEqual(lo, 0.2993, places=3)
        self.assertAlmostEqual(hi, 0.7007, places=3)
        self.assertAlmostEqual(stats.phi([1, 1, 0, 0], [1, 1, 0, 0]), 1.0)
        self.assertIsNone(stats.phi([1, 1], [1, 0]))


class EndToEnd(unittest.TestCase):
    def test_dry_run_shows_shared_memory_failure(self):
        from memcheck.cli import main
        with tempfile.TemporaryDirectory() as d:
            main(["--data", str(FIX), "run", "--backend", "dry", "--n", "0", "--k", "1", "--out", d, "--name", "t"])
            s = json.loads((Path(d) / "t" / "summary.json").read_text())
            self.assertEqual(s["n_records"], 6)
            inc = s["judge"]["retrieval_incomplete_non_adv"]
            # two stubs that are right only when memory holds the evidence agree on the same wrong answer otherwise
            self.assertGreater(inc["n_agree_wrong"], 0)
            self.assertEqual(s["judge"]["retrieval_full_non_adv"]["n_agree_wrong"], 0)
            # resume: a second call adds nothing
            main(["--data", str(FIX), "run", "--backend", "dry", "--n", "0", "--k", "1", "--out", d, "--name", "t"])
            self.assertEqual(len((Path(d) / "t" / "records.jsonl").read_text().splitlines()), 6)


if __name__ == "__main__":
    unittest.main()
