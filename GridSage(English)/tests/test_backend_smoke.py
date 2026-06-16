import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "backend" / "vgridsim_core"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CORE_PATH) not in sys.path:
    sys.path.insert(0, str(CORE_PATH))


class BackendSmokeTests(unittest.TestCase):
    def test_health_reports_solver_diagnostics(self):
        from fastapi.testclient import TestClient

        from backend.main import app

        client = TestClient(app)
        response = client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("python", payload)
        self.assertIn("packages", payload)
        self.assertIn("solver", payload)
        self.assertIn("full_baseline_ready", payload["solver"])

    def test_session_and_grid_snapshot_endpoint(self):
        from fastapi.testclient import TestClient

        from backend.main import app

        client = TestClient(app)
        session_response = client.post("/api/session/new")
        self.assertEqual(session_response.status_code, 200)
        session_id = session_response.json()["session_id"]

        snapshot_response = client.get(f"/api/grid_snapshot/{session_id}")
        self.assertEqual(snapshot_response.status_code, 200)
        snapshot = snapshot_response.json()
        self.assertEqual(snapshot["grid_model"], "ieee33")
        self.assertEqual(snapshot["device_summary"]["bus_count"], 33)
        self.assertEqual(len(snapshot["nodes"]), 33)
        self.assertGreaterEqual(len(snapshot["edges"]), 32)

    def test_apply_delta_normalizes_baseline_and_disabled_devices(self):
        from backend.main import apply_delta
        from backend.schema import DeltaCommand, ScenarioConfig

        state = ScenarioConfig(algo_name="PPO", execution_mode="train")
        updated = apply_delta(
            state,
            [
                DeltaCommand(action="update_config", target="algo_name", value="baseline"),
                DeltaCommand(
                    action="update_config",
                    target="disabled_devices",
                    value={"3": {"gen": True}},
                ),
            ],
        )

        self.assertEqual(updated.algo_name, "Baseline")
        self.assertEqual(updated.execution_mode, "evaluate")
        self.assertEqual(updated.disabled_devices, {"b3": {"generator": ["*"]}})

    def test_validation_rejects_unsafe_multiplier(self):
        from backend.schema import ScenarioConfig
        from backend.validation import validate_scenario

        ok, message = validate_scenario(ScenarioConfig(global_pv_multiplier=3.1))
        self.assertFalse(ok)
        self.assertIn("global_pv_multiplier", message)

    def test_skill_retriever_matches_high_pv_low_load(self):
        from backend.schema import ScenarioConfig
        from backend.skill_retriever import retrieve_skills

        matches, warnings = retrieve_skills("simulate high PV low load at midday", ScenarioConfig())
        self.assertTrue(matches)
        self.assertEqual(matches[0].skill_id, "S01_HIGH_PV_LOW_LOAD")
        self.assertTrue(warnings)

    def test_seg_func_helper_builds_callable_profile(self):
        from grid_model import _seg_func_from_series

        profile = _seg_func_from_series([0, 3600, 7200], [1.0, 2.0, 3.0])
        self.assertEqual(profile(0), 1.0)
        self.assertEqual(profile(3599), 1.0)
        self.assertEqual(profile(3600), 2.0)
        self.assertEqual(profile(9000), 3.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
