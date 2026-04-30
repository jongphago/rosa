import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "src" / "turtle_agent" / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from ko_query_router import preprocess_korean_query  # noqa: E402


class TestKoQueryRouter(unittest.TestCase):
    def test_korean_route_slots_abcd(self):
        result = preprocess_korean_query("turtle1을 A에서 B로 이동해줘")
        self.assertEqual(result["intent"], "navigate")
        self.assertGreaterEqual(float(result["confidence"]), 0.8)
        self.assertEqual(result["slots"].get("from"), "A")
        self.assertEqual(result["slots"].get("to"), "B")
        self.assertEqual(result["slots"].get("turtle"), "turtle1")
        self.assertTrue(result["preprocessing_block"])

    def test_korean_route_slots_coordinates(self):
        result = preprocess_korean_query("(2,9)에서 (9,9)로 이동")
        self.assertEqual(result["intent"], "navigate")
        self.assertEqual(result["slots"].get("from_xy"), (2.0, 9.0))
        self.assertEqual(result["slots"].get("to_xy"), (9.0, 9.0))

    def test_reset_query(self):
        result = preprocess_korean_query("reset")
        self.assertEqual(result["intent"], "reset")
        self.assertEqual(float(result["confidence"]), 1.0)
        self.assertFalse(result["preprocessing_block"])


if __name__ == "__main__":
    unittest.main()

