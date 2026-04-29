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

import json
import sys
import time
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "src" / "turtle_agent" / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from navigation_progress import NavigationGoalStore  # noqa: E402
from obstacle_store import AabbGeometry, Obstacle, ObstacleStore  # noqa: E402
from tools import navigation as navigation_tools  # noqa: E402


class FakePoseTool:
    def __init__(self, poses):
        self._poses = list(poses)

    def read(self, name):
        _ = name
        return self._poses.pop(0)


class TestNavigationTools(unittest.TestCase):
    def setUp(self):
        self.store = ObstacleStore()
        navigation_tools.configure_navigation_context(self.store, NavigationGoalStore())

    def test_set_and_evaluate_goal_progress(self):
        fake_pose = FakePoseTool([(1.0, 0.0), (3.0, 0.0)])
        navigation_tools.configure_navigation_context(
            self.store,
            NavigationGoalStore(),
            pose_reader=fake_pose.read,
        )

        set_result = json.loads(
            navigation_tools.set_navigation_goal.invoke(
                {
                    "name": "turtle1",
                    "goal_x": 7.0,
                    "goal_y": 0.0,
                }
            )
        )
        eval_result = json.loads(
            navigation_tools.evaluate_navigation_progress.invoke({"name": "turtle1"})
        )

        self.assertEqual(set_result["status"], "ok")
        self.assertEqual(set_result["assessment"], "goal_set")
        self.assertEqual(eval_result["status"], "ok")
        self.assertEqual(eval_result["assessment"], "good_progress")
        self.assertAlmostEqual(eval_result["goal_distance_delta"], -2.0)

    def test_set_goal_records_active_obstacle(self):
        self.store.upsert(
            Obstacle(
                id="wet",
                kind="temporary",
                geometry=AabbGeometry(min_x=4.0, min_y=-1.0, max_x=5.0, max_y=1.0),
                expires_at=time.monotonic() + 60.0,
            )
        )
        fake_pose = FakePoseTool([(1.0, 0.0)])
        navigation_tools.configure_navigation_context(
            self.store,
            NavigationGoalStore(),
            pose_reader=fake_pose.read,
        )

        result = json.loads(
            navigation_tools.set_navigation_goal.invoke(
                {
                    "name": "turtle1",
                    "goal_x": 7.0,
                    "goal_y": 0.0,
                }
            )
        )

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["goal"]["obstacle_active"])
        self.assertEqual(result["goal"]["active_obstacles"], ["wet"])

    def test_evaluate_without_goal_returns_structured_missing_goal(self):
        result = json.loads(
            navigation_tools.evaluate_navigation_progress.invoke({"name": "turtle1"})
        )

        self.assertEqual(result["status"], "missing_goal")
        self.assertEqual(result["assessment"], "not_configured")
        self.assertEqual(result["recommendation"], "set_navigation_goal")


if __name__ == "__main__":
    unittest.main()
