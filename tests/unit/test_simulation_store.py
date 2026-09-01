"""
Unit tests for SimulationStore - persistence of simulation / what-if runs.
"""

import tempfile
import unittest
from pathlib import Path

from src.portfolio.simulation_store import SimulationStore


class TestSimulationStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = self._tmp.name
        self.store = SimulationStore(self.data_dir)

    def tearDown(self):
        self._tmp.cleanup()

    def test_init_creates_simulations_dir(self):
        self.assertTrue((Path(self.data_dir) / "simulations").is_dir())

    def test_save_returns_well_formed_id(self):
        sim_id = self.store.save("advanced_what_if", {"a": 1}, "output")
        self.assertTrue(sim_id.startswith("sim_"))
        # id must be path-traversal safe (alphanumeric + underscore only)
        self.assertTrue(SimulationStore._is_valid_id(sim_id))
        self.assertTrue((Path(self.data_dir) / "simulations" / f"{sim_id}.json").exists())

    def test_save_and_get_roundtrip(self):
        sim_id = self.store.save(
            tool="advanced_what_if",
            inputs={"scenario_type": "likely", "random_seed": 7},
            output="🔮 result body",
            portfolio_id="pid-1",
            portfolio_name="My Portfolio",
            random_seed=7,
        )
        rec = self.store.get(sim_id)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["id"], sim_id)
        self.assertEqual(rec["tool"], "advanced_what_if")
        self.assertEqual(rec["output"], "🔮 result body")
        self.assertEqual(rec["portfolio_id"], "pid-1")
        self.assertEqual(rec["portfolio_name"], "My Portfolio")
        self.assertEqual(rec["random_seed"], 7)
        self.assertEqual(rec["inputs"], {"scenario_type": "likely", "random_seed": 7})
        self.assertIn("created_at", rec)

    def test_get_missing_returns_none(self):
        self.assertIsNone(self.store.get("sim_does_not_exist"))

    def test_get_rejects_path_traversal(self):
        self.assertIsNone(self.store.get("../secrets"))
        self.assertIsNone(self.store.get("a/b"))
        self.assertIsNone(self.store.get(""))

    def test_list_returns_summaries_without_output(self):
        self.store.save("advanced_what_if", {"a": 1}, "big output", portfolio_id="p1")
        summaries = self.store.list()
        self.assertEqual(len(summaries), 1)
        self.assertNotIn("output", summaries[0])
        self.assertIn("inputs", summaries[0])
        self.assertEqual(summaries[0]["tool"], "advanced_what_if")

    def test_list_newest_first(self):
        # created_at is second-resolution, so set it explicitly to guarantee ordering
        ids = []
        for i, ts in enumerate(["2026-01-01T00:00:00", "2026-06-01T00:00:00", "2026-03-01T00:00:00"]):
            sim_id = self.store.save("optimize_portfolio", {"i": i}, f"out{i}")
            rec = self.store.get(sim_id)
            rec["created_at"] = ts
            (Path(self.data_dir) / "simulations" / f"{sim_id}.json").write_text(
                __import__("json").dumps(rec)
            )
            ids.append((sim_id, ts))
        ordered = [s["created_at"] for s in self.store.list()]
        self.assertEqual(ordered, ["2026-06-01T00:00:00", "2026-03-01T00:00:00", "2026-01-01T00:00:00"])

    def test_list_filters_by_portfolio(self):
        self.store.save("advanced_what_if", {}, "a", portfolio_id="p1")
        self.store.save("advanced_what_if", {}, "b", portfolio_id="p2")
        self.assertEqual(len(self.store.list(portfolio_id="p1")), 1)
        self.assertEqual(len(self.store.list(portfolio_id="p2")), 1)
        self.assertEqual(len(self.store.list()), 2)

    def test_list_filters_by_tool(self):
        self.store.save("advanced_what_if", {}, "a")
        self.store.save("optimize_portfolio", {}, "b")
        result = self.store.list(tool="optimize_portfolio")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["tool"], "optimize_portfolio")

    def test_list_respects_limit(self):
        for i in range(5):
            self.store.save("advanced_what_if", {"i": i}, f"out{i}")
        self.assertEqual(len(self.store.list(limit=3)), 3)

    def test_list_ignores_unrelated_files(self):
        # A stray non-sim file in the directory should not break listing
        (Path(self.data_dir) / "simulations" / "notes.txt").write_text("hello")
        self.store.save("advanced_what_if", {}, "a")
        self.assertEqual(len(self.store.list()), 1)

    def test_delete(self):
        sim_id = self.store.save("advanced_what_if", {}, "a")
        self.assertTrue(self.store.delete(sim_id))
        self.assertIsNone(self.store.get(sim_id))
        # deleting again is a no-op returning False
        self.assertFalse(self.store.delete(sim_id))

    def test_delete_rejects_path_traversal(self):
        self.assertFalse(self.store.delete("../x"))

    def test_none_seed_persisted(self):
        sim_id = self.store.save("simulate_what_if", {"start": "2026-01-01"}, "out")
        rec = self.store.get(sim_id)
        self.assertIsNone(rec["random_seed"])


if __name__ == "__main__":
    unittest.main()
