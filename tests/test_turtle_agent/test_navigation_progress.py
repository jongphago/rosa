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

import sys
import time
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "src" / "turtle_agent" / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from navigation_progress import (  # noqa: E402
    evaluate_navigation_goal,
    make_navigation_goal,
    obstacle_clearance,
)
from obstacle_store import AabbGeometry, Obstacle  # noqa: E402


def _temporary_aabb() -> Obstacle:
    return Obstacle(
        id="wet",
        kind="temporary",
        geometry=AabbGeometry(min_x=4.0, min_y=-1.0, max_x=5.0, max_y=1.0),
        expires_at=time.monotonic() + 60.0,
    )


class TestNavigationProgress(unittest.TestCase):
    def test_goal_distance_decrease_is_good_progress(self):
        goal = make_navigation_goal(
            turtle="turtle1",
            start=(1.0, 0.0),
            final_goal=(7.0, 0.0),
            obstacles=[],
        )

        updated, result = evaluate_navigation_goal(
            goal=goal,
            current=(3.0, 0.0),
            obstacles=[],
        )

        self.assertEqual(result["assessment"], "good_progress")
        self.assertEqual(result["recommendation"], "continue")
        self.assertAlmostEqual(result["goal_distance_delta"], -2.0)
        self.assertAlmostEqual(updated.previous_goal_distance, 4.0)

    def test_goal_distance_increase_without_obstacle_is_bad_regression(self):
        goal = make_navigation_goal(
            turtle="turtle1",
            start=(1.0, 0.0),
            final_goal=(7.0, 0.0),
            obstacles=[],
        )

        _, result = evaluate_navigation_goal(
            goal=goal,
            current=(0.0, 0.0),
            obstacles=[],
        )

        self.assertEqual(result["assessment"], "bad_regression")
        self.assertEqual(result["recommendation"], "stop_and_replan")
        self.assertAlmostEqual(result["goal_distance_delta"], 1.0)

    def test_obstacle_clearance_increase_allows_avoidance_regression(self):
        obstacle = _temporary_aabb()
        goal = make_navigation_goal(
            turtle="turtle1",
            start=(1.0, 0.0),
            final_goal=(7.0, 0.0),
            obstacles=[obstacle],
            turtle_radius=0.5,
            safety_margin=0.2,
        )

        self.assertEqual(goal.active_obstacles, ("wet",))
        _, result = evaluate_navigation_goal(
            goal=goal,
            current=(0.0, 0.0),
            obstacles=[obstacle],
        )

        self.assertEqual(result["assessment"], "good_avoidance")
        self.assertEqual(result["recommendation"], "continue_avoidance")
        self.assertGreater(result["obstacle_distance_deltas"]["wet"], 0.0)

    def test_aabb_clearance_is_negative_inside_obstacle(self):
        obstacle = _temporary_aabb()
        self.assertLess(obstacle_clearance((4.5, 0.0), obstacle, 0.5), 0.0)
        self.assertAlmostEqual(obstacle_clearance((6.0, 0.0), obstacle, 0.5), 0.5)


if __name__ == "__main__":
    unittest.main()
