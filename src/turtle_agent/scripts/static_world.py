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

"""Load and optionally draw static obstacles into a shared ObstacleStore."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from obstacle_store import ObstacleStore
from static_map_loader import (
    StaticMapLoadError,
    load_into_store,
    obstacles_from_data_for_visual,
    parse_map_file,
)
from world_builder import draw_static_world


def _read_initial_turtles(path: str) -> List[Dict[str, Any]]:
    """Read optional ``initial_turtles`` list from YAML/JSON config."""
    p = Path(path)
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text)
    else:
        return []
    if not isinstance(data, dict):
        return []
    items = data.get("initial_turtles", [])
    if not isinstance(items, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).replace("/", "").strip()
        if not name:
            continue
        try:
            x = float(item.get("x"))
            y = float(item.get("y"))
            theta = float(item.get("theta", 0.0))
        except (TypeError, ValueError):
            continue
        out.append({"name": name, "x": x, "y": y, "theta": theta})
    return out


def _spawn_initial_turtles(specs: List[Dict[str, Any]]) -> None:
    """Spawn turtles from config to fixed initial points.

    Existing turtles with the same names are removed first. This allows
    overriding the default center-spawned ``turtle1`` as well.
    """
    if not specs:
        return
    import rospy
    from turtlesim.srv import Kill, Spawn

    rospy.wait_for_service("/spawn", timeout=5.0)
    rospy.wait_for_service("/kill", timeout=5.0)
    spawn = rospy.ServiceProxy("/spawn", Spawn)
    kill = rospy.ServiceProxy("/kill", Kill)

    for spec in specs:
        name = spec["name"]
        x = spec["x"]
        y = spec["y"]
        theta = spec["theta"]
        try:
            try:
                kill(name)
            except Exception:
                pass
            spawn(x=x, y=y, theta=theta, name=name)
        except Exception as e:
            rospy.logwarn("initial turtle spawn skipped for %s: %s", name, e)


def load_static_world(obstacle_store: ObstacleStore) -> None:
    """Load configured static obstacles and optionally draw them in turtlesim."""
    import rospy

    path = str(rospy.get_param("~static_obstacles_file", "")).strip()
    if path:
        try:
            p = Path(path).expanduser()
            data = parse_map_file(p)
            load_into_store(obstacle_store, data, source=str(p))
        except StaticMapLoadError as e:
            rospy.logerr("static obstacles: %s", e)
            raise
        if rospy.get_param("~draw_static_world", True):
            try:
                draw_shapes = obstacles_from_data_for_visual(data, source=str(p))
                count = draw_static_world(
                    obstacle_store, draw_obstacles=draw_shapes
                )
                rospy.loginfo("static world builder drew %s segments", count)
            except Exception as e:
                rospy.logerr("static world builder failed: %s", e)
                if rospy.get_param("~world_builder_required", True):
                    raise
        try:
            initial_turtles = _read_initial_turtles(path)
            _spawn_initial_turtles(initial_turtles)
            if initial_turtles:
                rospy.loginfo(
                    "spawned initial turtles from map config: %s",
                    ", ".join(t["name"] for t in initial_turtles),
                )
        except Exception as e:
            rospy.logwarn("initial turtle spawn skipped: %s", e)
