#  Copyright (c) 2024. Jet Propulsion Laboratory. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""LangChain tools for turtle navigation-goal progress evaluation."""

from __future__ import annotations

import json
from typing import Any, Callable, Iterable, Optional, Tuple

from langchain.agents import tool
from navigation_progress import (
    NavigationGoalStore,
    evaluate_navigation_goal,
    goal_to_record,
    make_navigation_goal,
)
from obstacle_store import ObstacleStore

_OBSTACLE_STORE: Optional[ObstacleStore] = None
_GOAL_STORE = NavigationGoalStore()
_POSE_READER: Optional[Callable[[str], Tuple[float, float]]] = None


class NavigationToolError(ValueError):
    """Invalid navigation-tool input or missing process context."""


def configure_navigation_context(
    obstacle_store: ObstacleStore,
    goal_store: Optional[NavigationGoalStore] = None,
    pose_reader: Optional[Callable[[str], Tuple[float, float]]] = None,
) -> None:
    """Inject shared stores used by navigation progress tools."""
    global _OBSTACLE_STORE, _GOAL_STORE, _POSE_READER
    _OBSTACLE_STORE = obstacle_store
    if goal_store is not None:
        _GOAL_STORE = goal_store
    _POSE_READER = pose_reader


def get_configured_goal_store() -> NavigationGoalStore:
    return _GOAL_STORE


def _require_obstacle_store() -> ObstacleStore:
    if _OBSTACLE_STORE is None:
        raise NavigationToolError("ObstacleStore is not configured.")
    return _OBSTACLE_STORE


def _coerce_float(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NavigationToolError(f"{key} must be a number.")
    return float(value)


def _normalize_turtle_name(name: str) -> str:
    out = str(name).replace("/", "").strip()
    if not out:
        raise NavigationToolError("name must be a non-empty turtle name.")
    return out


def _parse_kinds(obstacle_kinds: str) -> Tuple[str, ...]:
    kinds = tuple(item.strip().lower() for item in str(obstacle_kinds).split(",") if item.strip())
    if not kinds:
        raise NavigationToolError("obstacle_kinds must include at least one kind.")
    return kinds


def _pose_xy(name: str) -> Tuple[float, float]:
    if _POSE_READER is not None:
        x, y = _POSE_READER(name)
        return float(x), float(y)
    import tools.turtle as turtle_tools

    poses = turtle_tools.get_turtle_pose.invoke({"names": [name]})
    if isinstance(poses, dict) and "Error" in poses:
        raise NavigationToolError(str(poses["Error"]))
    try:
        pose = poses[name]
        return float(getattr(pose, "x")), float(getattr(pose, "y"))
    except Exception as e:
        raise NavigationToolError(f"Failed to read pose for {name}: {e}") from e


def _error_payload(message: str, *, assessment: str = "error") -> str:
    return json.dumps(
        {
            "status": "error",
            "assessment": assessment,
            "recommendation": "fix_input_or_configure_context",
            "reason": message,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _obstacle_ids(obstacles: Iterable[Any]) -> list[str]:
    return [str(getattr(obstacle, "id", "")) for obstacle in obstacles]


@tool
def set_navigation_goal(
    name: str,
    goal_x: float,
    goal_y: float,
    turtle_radius: float = 0.5,
    safety_margin: float = 0.2,
    goal_tolerance: float = 0.2,
    obstacle_kinds: str = "temporary",
) -> str:
    """
    Store the final navigation goal for a turtle and identify direct-path obstacles.

    Call this before multi-step navigation when a turtle's final destination is
    known. Later movement chunks can be evaluated against the same final goal.
    """
    try:
        turtle_name = _normalize_turtle_name(name)
        current = _pose_xy(turtle_name)
        final_goal = (
            _coerce_float(goal_x, "goal_x"),
            _coerce_float(goal_y, "goal_y"),
        )
        obstacles = _require_obstacle_store().snapshot()
        goal = make_navigation_goal(
            turtle=turtle_name,
            start=current,
            final_goal=final_goal,
            obstacles=obstacles,
            turtle_radius=_coerce_float(turtle_radius, "turtle_radius"),
            safety_margin=_coerce_float(safety_margin, "safety_margin"),
            goal_tolerance=_coerce_float(goal_tolerance, "goal_tolerance"),
            obstacle_kinds=_parse_kinds(obstacle_kinds),
        )
        _GOAL_STORE.upsert(goal)
        return json.dumps(
            {
                "status": "ok",
                "goal": goal_to_record(goal),
                "checked_obstacles": _obstacle_ids(obstacles),
                "assessment": "goal_set",
                "recommendation": "move_then_evaluate_progress",
                "reason": "터틀별 최종 도착지를 저장함",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    except (NavigationToolError, ValueError, TypeError) as e:
        return _error_payload(str(e))


@tool
def evaluate_navigation_progress(name: str, progress_epsilon: float = 0.01) -> str:
    """
    Evaluate whether a turtle's current pose is progressing toward its final goal.

    The result is deterministic JSON and includes assessment, recommendation,
    and reason fields for the agent's next tool-call decision.
    """
    try:
        turtle_name = _normalize_turtle_name(name)
        goal = _GOAL_STORE.get(turtle_name)
        if goal is None:
            return json.dumps(
                {
                    "status": "missing_goal",
                    "turtle": turtle_name,
                    "assessment": "not_configured",
                    "recommendation": "set_navigation_goal",
                    "reason": "해당 터틀의 최종 도착지가 설정되어 있지 않음",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        current = _pose_xy(turtle_name)
        updated_goal, result = evaluate_navigation_goal(
            goal=goal,
            current=current,
            obstacles=_require_obstacle_store().snapshot(),
            progress_epsilon=_coerce_float(progress_epsilon, "progress_epsilon"),
        )
        _GOAL_STORE.upsert(updated_goal)
        result["status"] = "ok"
        return json.dumps(result, ensure_ascii=False, sort_keys=True)
    except (NavigationToolError, ValueError, TypeError) as e:
        return _error_payload(str(e))


@tool
def clear_navigation_goal(name: str = "") -> str:
    """Clear one turtle's navigation goal, or all goals when name is empty."""
    try:
        turtle_name = str(name).replace("/", "").strip()
        if turtle_name:
            removed = _GOAL_STORE.remove(turtle_name)
            payload = {
                "status": "ok",
                "cleared": removed,
                "turtle": turtle_name,
                "assessment": "goal_cleared" if removed else "goal_not_found",
                "recommendation": "set_navigation_goal",
                "reason": "터틀의 최종 도착지 상태를 삭제함"
                if removed
                else "삭제할 최종 도착지 상태가 없음",
            }
        else:
            _GOAL_STORE.clear()
            payload = {
                "status": "ok",
                "cleared": True,
                "assessment": "all_goals_cleared",
                "recommendation": "set_navigation_goal",
                "reason": "모든 터틀의 최종 도착지 상태를 삭제함",
            }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except Exception as e:
        return _error_payload(str(e))
