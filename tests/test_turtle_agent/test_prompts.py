import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "src" / "turtle_agent" / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from prompts import get_prompts  # noqa: E402


class TestTurtleAgentPrompts(unittest.TestCase):
    def test_system_prompt_does_not_default_to_obstacle_strategy(self):
        prompts = get_prompts()
        combined = "\n".join(
            str(getattr(prompts, field, ""))
            for field in (
                "critical_instructions",
                "constraints_and_guardrails",
                "about_your_capabilities",
            )
        )

        self.assertNotIn("list/check obstacles first", combined)
        self.assertNotIn("check_path_against_obstacles before movement", combined)
        self.assertNotIn("Avoid single long direct moves", combined)
        self.assertNotIn("Move in short segments", combined)


if __name__ == "__main__":
    unittest.main()
