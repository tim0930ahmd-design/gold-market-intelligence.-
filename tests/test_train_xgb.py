import ast
from pathlib import Path
import unittest

SOURCE = Path(__file__).parents[1] / "train_xgb.py"
TEXT = SOURCE.read_text(encoding="utf-8")
TREE = ast.parse(TEXT)
CONSTANTS = {}
for node in TREE.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                try:
                    CONSTANTS[target.id] = ast.literal_eval(node.value)
                except Exception:
                    pass
FUNCTIONS = {node.name for node in TREE.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}

class TrainingPipelineContractTests(unittest.TestCase):
    def test_exact_engine_feature_order(self):
        expected = ["qualityScore","stopAtr","rr","regimeConfidence","atrRatio","efficiency","bodyAtr","emaAlignment","slope","spreadAtr","isBuy","expertSweep","expertBreakout","expertRetest","expertContinuation","expertReversal","expertPullback","expertRange","expertMomentum","expertVision","regimeTrend","regimeRange","regimeCompression","regimeExpansion","regimeImpulse"]
        self.assertEqual(CONSTANTS["META_FEATURES"], expected)
        self.assertEqual(len(CONSTANTS["META_FEATURES"]), 25)

    def test_atr_first_touch_contract(self):
        self.assertEqual(CONSTANTS["HORIZON"], 12)
        self.assertEqual(CONSTANTS["STOP_ATR"], 1.0)
        self.assertEqual(CONSTANTS["RR"], 1.8)
        self.assertIn('if hs:label="STOP"', TEXT)

    def test_temporal_validation_and_separate_models_exist(self):
        self.assertIn("purge_delta=pd.Timedelta(minutes=5*HORIZON)", TEXT)
        self.assertIn("modelsBySide", TEXT)
        self.assertIn("runtime_context_parity", TEXT)
        self.assertIn("oos_combined_037", TEXT)
        self.assertIn("oos_combined_074", TEXT)

    def test_required_pipeline_functions_exist(self):
        for name in ("load_bars","indicators","regime_at","candidate_rows","build_samples","fit_models","evaluate","tree_json","main"):
            self.assertIn(name, FUNCTIONS)

if __name__ == "__main__":
    unittest.main()
