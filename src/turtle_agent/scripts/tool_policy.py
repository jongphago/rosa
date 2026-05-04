from __future__ import annotations

from dataclasses import dataclass
from typing import Any, FrozenSet, Iterable, Tuple

OBSTACLE_VALIDATION_TAG = "obstacle_validation"


@dataclass(frozen=True)
class ToolPolicyEvidence:
    enabled_tags: FrozenSet[str] = frozenset()
    source_records: Tuple[str, ...] = ()
    reasons: Tuple[str, ...] = ()

    @classmethod
    def from_memory_result(cls, memory_result: Any) -> "ToolPolicyEvidence":
        if memory_result is None:
            return cls()
        tags = frozenset(
            str(tag) for tag in getattr(memory_result, "policy_tags", ()) if tag
        )
        records = []
        for row in getattr(memory_result, "selected_records", ()) or ():
            if not isinstance(row, dict):
                continue
            meta = row.get("meta", {})
            if not isinstance(meta, dict):
                meta = {}
            record_id = (
                row.get("record_id")
                or row.get("id")
                or meta.get("session_id")
            )
            if record_id:
                records.append(str(record_id))
        reasons = tuple(
            str(reason)
            for reason in getattr(memory_result, "policy_reasons", ())
            if reason
        )
        return cls(enabled_tags=tags, source_records=tuple(records), reasons=reasons)


@dataclass(frozen=True)
class ToolPolicyRequirement:
    required_tags: FrozenSet[str]
    reason: str


@dataclass(frozen=True)
class ToolPolicyDecision:
    active_tools: Tuple[Any, ...]
    disabled_tools: Tuple[str, ...]
    enabled_tags: Tuple[str, ...]


TOOL_POLICY_REQUIREMENTS = {
    "check_path_against_obstacles": ToolPolicyRequirement(
        required_tags=frozenset({OBSTACLE_VALIDATION_TAG}),
        reason="candidate segment validation requires selected obstacle-related memory",
    )
}


def _tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", "") or getattr(tool, "__name__", ""))


def filter_tools_for_policy(
    tools: Iterable[Any], evidence: ToolPolicyEvidence
) -> ToolPolicyDecision:
    active = []
    disabled = []
    enabled_tags = set(evidence.enabled_tags)
    for tool in tools:
        name = _tool_name(tool)
        requirement = TOOL_POLICY_REQUIREMENTS.get(name)
        if requirement and not requirement.required_tags.issubset(enabled_tags):
            disabled.append(name)
            continue
        active.append(tool)
    return ToolPolicyDecision(
        active_tools=tuple(active),
        disabled_tools=tuple(disabled),
        enabled_tags=tuple(sorted(enabled_tags)),
    )
