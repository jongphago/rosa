import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "src" / "turtle_agent" / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from tool_policy import (  # noqa: E402
    OBSTACLE_VALIDATION_TAG,
    ToolPolicyEvidence,
    filter_tools_for_policy,
)


class TestToolPolicy(unittest.TestCase):
    def test_policy_evidence_from_memory_result(self):
        memory_result = SimpleNamespace(
            policy_tags=("obstacle_validation",),
            selected_records=({"record_id": "rec-1"},),
            policy_reasons=("collision evidence",),
        )

        evidence = ToolPolicyEvidence.from_memory_result(memory_result)

        self.assertEqual(evidence.enabled_tags, frozenset({"obstacle_validation"}))
        self.assertEqual(evidence.source_records, ("rec-1",))
        self.assertEqual(evidence.reasons, ("collision evidence",))

    def test_policy_sensitive_tool_is_hidden_without_evidence(self):
        tools = [
            SimpleNamespace(name="get_turtle_pose"),
            SimpleNamespace(name="check_path_against_obstacles"),
        ]

        decision = filter_tools_for_policy(tools, ToolPolicyEvidence())

        self.assertEqual([tool.name for tool in decision.active_tools], ["get_turtle_pose"])
        self.assertEqual(decision.disabled_tools, ("check_path_against_obstacles",))

    def test_policy_sensitive_tool_is_visible_with_required_evidence(self):
        tools = [
            SimpleNamespace(name="get_turtle_pose"),
            SimpleNamespace(name="check_path_against_obstacles"),
        ]
        evidence = ToolPolicyEvidence(enabled_tags=frozenset({OBSTACLE_VALIDATION_TAG}))

        decision = filter_tools_for_policy(tools, evidence)

        self.assertEqual(
            [tool.name for tool in decision.active_tools],
            ["get_turtle_pose", "check_path_against_obstacles"],
        )
        self.assertEqual(decision.disabled_tools, ())

    def test_unregistered_tools_remain_visible_by_default(self):
        tools = [
            SimpleNamespace(name="list_obstacles"),
            SimpleNamespace(name="publish_twist_to_cmd_vel"),
        ]

        decision = filter_tools_for_policy(tools, ToolPolicyEvidence())

        self.assertEqual(
            [tool.name for tool in decision.active_tools],
            ["list_obstacles", "publish_twist_to_cmd_vel"],
        )


if __name__ == "__main__":
    unittest.main()
