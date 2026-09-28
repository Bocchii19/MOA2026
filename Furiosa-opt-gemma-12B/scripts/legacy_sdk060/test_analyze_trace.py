"""Regression checks for overlapping spans, independent of the frozen grader."""

import unittest

from analyze_trace import duration, intersection_duration, parse, summarize


class TraceTests(unittest.TestCase):
    def test_nested_duplicate_and_adjacent_intervals(self):
        self.assertEqual(duration([(0, 10), (2, 3), (0, 10), (10, 12), (20, 20)]), 12)
        self.assertEqual(duration([]), 0)

    def test_real_k2_wait_overlap(self):
        # Arena 72093: summing these waits incorrectly reports 14,740 residual cycles.
        waits = [(29520, 46129), (30360, 50122), (51108, 51967)]
        self.assertEqual(duration(waits), 21461)
        self.assertEqual(51970 - duration(waits), 30509)
        self.assertEqual(intersection_duration(waits, [(46130, 47938)]), 1808)

    def test_repeated_kernel_invocations_stay_separate(self):
        text = ("==> k\n    cycles=12\n"
                "    SPAN Task begin=0 end=12 dur=12\n"
                "SPAN Cluster begin=3 end=9 dur=6\n"
                "SPAN DMA begin=7 end=11 dur=4\n"
                "==> k\nSPAN Task begin=0 end=7 dur=7\n")
        runs = parse(text)
        self.assertEqual(len(runs), 2)
        result = summarize(runs[0])
        self.assertEqual(result["reported_cycles"], 12)
        self.assertEqual(result["hull_minus_cluster_union_cycles"], 6)
        self.assertEqual(result["cluster_engine_overlap_cycles"], 2)
        self.assertIsNone(summarize(runs[1])["reported_cycles"])

    def test_invalid_spans_fail_loudly(self):
        for text in ("SPAN Task begin=0 end=1 dur=1",
                     "==> k\nSPAN Task begin=0 end=1 dur=2",
                     "==> k\nSPAN Task broken"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse(text)


if __name__ == "__main__":
    unittest.main()
