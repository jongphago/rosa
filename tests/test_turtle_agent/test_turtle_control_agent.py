#  Copyright (c) 2024. Jet Propulsion Laboratory. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.

import sys
import threading
import time
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "src" / "turtle_agent" / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from turtle_control_agent import (  # noqa: E402
    DEFAULT_WORKER_SYSTEM_PROMPT,
    TurtleControlAgent,
    TurtleTask,
)
from turtle_control_prompts import WORKER_SYSTEM_PROMPT  # noqa: E402


def _plan_segment_prompts(_user_prompt, count):
    return [
        f"선분 {index}을 그리시오"
        for index in range(count)
    ]


def _make_recording_worker(log, lock, delay=0.0, fail_first=False):
    calls = {"count": 0}

    def worker(task):
        with lock:
            calls["count"] += 1
            call_number = calls["count"]
            log.append((task.turtle_id, task.instruction, time.monotonic()))
        if fail_first and call_number == 1:
            raise RuntimeError(f"{task.turtle_id} planned failure")
        if delay:
            time.sleep(delay)
        return f"{task.turtle_id}: received {task.instruction}"

    return worker


class FakeLlmPlanner:
    def __init__(self, prompts):
        self.prompts = tuple(prompts)
        self.calls = []

    def invoke(self, user_prompt):
        self.calls.append(user_prompt)
        return self.prompts


class TestTurtleControlAgent(unittest.TestCase):
    def test_default_worker_system_prompt_matches_task_policy(self):
        self.assertEqual(DEFAULT_WORKER_SYSTEM_PROMPT, WORKER_SYSTEM_PROMPT)
        self.assertIn("worker agent", DEFAULT_WORKER_SYSTEM_PROMPT)
        self.assertIn("전달받은 task 하나만 수행", DEFAULT_WORKER_SYSTEM_PROMPT)
        self.assertIn("결과만 간결하게 답하세요", DEFAULT_WORKER_SYSTEM_PROMPT)

    def test_runs_workers_in_parallel(self):
        barrier = threading.Barrier(2)
        starts = []
        lock = threading.Lock()

        def make_worker(label):
            def worker(task):
                with lock:
                    starts.append((label, time.monotonic()))
                barrier.wait(timeout=1.0)
                time.sleep(0.02)
                return f"{task.turtle_id}:{task.instruction}"

            return worker

        control = TurtleControlAgent(
            {
                "turtle1": make_worker("turtle1"),
                "turtle2": make_worker("turtle2"),
            }
        )

        results = control.run_parallel(
            [
                TurtleTask("turtle1", "draw left edge"),
                TurtleTask("turtle2", "draw right edge"),
            ]
        )

        self.assertEqual([r.ok for r in results], [True, True])
        self.assertEqual(results[0].output, "turtle1:draw left edge")
        self.assertEqual(results[1].output, "turtle2:draw right edge")
        self.assertEqual({name for name, _ in starts}, {"turtle1", "turtle2"})
        self.assertLess(abs(starts[0][1] - starts[1][1]), 0.2)

    def test_worker_failure_is_isolated(self):
        def ok_worker(task):
            return f"done:{task.instruction}"

        def failing_worker(_task):
            raise RuntimeError("boom")

        control = TurtleControlAgent(
            {
                "turtle1": failing_worker,
                "turtle2": ok_worker,
            }
        )

        results = control.run_parallel(
            [
                TurtleTask("turtle1", "bad task"),
                TurtleTask("turtle2", "good task"),
            ]
        )

        self.assertFalse(results[0].ok)
        self.assertIn("RuntimeError: boom", results[0].error)
        self.assertTrue(results[1].ok)
        self.assertEqual(results[1].output, "done:good task")

    def test_unknown_turtle_returns_failed_result(self):
        control = TurtleControlAgent({"turtle1": lambda task: task.instruction})

        results = control.run_parallel(
            [
                TurtleTask("turtle3", "unassigned"),
                TurtleTask("turtle1", "assigned"),
            ]
        )

        self.assertFalse(results[0].ok)
        self.assertIn("no worker registered", results[0].error)
        self.assertTrue(results[1].ok)
        self.assertEqual(results[1].output, "assigned")

    def test_add_worker_registers_new_turtle_for_future_tasks(self):
        control = TurtleControlAgent({"turtle1": lambda task: f"one:{task.instruction}"})

        control.add_worker("turtle2", lambda task: f"two:{task.instruction}")

        self.assertEqual(control.worker_ids(), ("turtle1", "turtle2"))
        results = control.run_parallel([TurtleTask("turtle2", "draw circle")])
        self.assertTrue(results[0].ok)
        self.assertEqual(results[0].output, "two:draw circle")

    def test_remove_worker_unassigns_turtle_for_future_tasks(self):
        control = TurtleControlAgent(
            {
                "turtle1": lambda task: f"one:{task.instruction}",
                "turtle2": lambda task: f"two:{task.instruction}",
            }
        )

        self.assertTrue(control.remove_worker("turtle2"))
        self.assertFalse(control.remove_worker("turtle2"))

        results = control.run_parallel([TurtleTask("turtle2", "draw circle")])
        self.assertFalse(results[0].ok)
        self.assertIn("no worker registered", results[0].error)

    def test_timeout_marks_unfinished_tasks(self):
        started = threading.Event()

        def slow_worker(_task):
            started.set()
            time.sleep(0.2)
            return "late"

        control = TurtleControlAgent({"turtle1": slow_worker})

        results = control.run_parallel([TurtleTask("turtle1", "slow")], timeout=0.01)

        self.assertTrue(started.is_set())
        self.assertFalse(results[0].ok)
        self.assertEqual(results[0].error, "task timed out")

    def test_plans_segment_prompts_from_user_prompt(self):
        prompts = _plan_segment_prompts("정사각형의 각 변을 그리시오", count=4)

        self.assertEqual(
            prompts,
            [
                "선분 0을 그리시오",
                "선분 1을 그리시오",
                "선분 2을 그리시오",
                "선분 3을 그리시오",
            ],
        )

    def test_normalizes_llm_planned_prompt_text(self):
        planned = TurtleControlAgent._normalize_planned_prompts(
            "- 왼쪽 변을 그리시오\n1. 오른쪽 변을 그리시오"
        )

        self.assertEqual(
            planned,
            (
                "왼쪽 변을 그리시오",
                "오른쪽 변을 그리시오",
            ),
        )

    def test_prompt_queue_assigns_next_prompt_to_completed_agent(self):
        log = []
        output_lines = []
        lock = threading.Lock()
        control = TurtleControlAgent(
            {
                "turtle1": _make_recording_worker(log, lock, delay=0.005),
                "turtle2": _make_recording_worker(log, lock, delay=0.05),
            },
            log_enabled=True,
            log_sink=output_lines.append,
        )
        prompts = _plan_segment_prompts("도형을 선분으로 나누어 그리시오", count=5)

        results = control.run_prompt_queue(prompts)

        self.assertEqual(len(results), 5)
        self.assertTrue(all(result.ok for result in results))
        outputs = {result.instruction: result.output for result in results}
        self.assertIn("received 선분 0을 그리시오", outputs["선분 0을 그리시오"])
        self.assertIn("received 선분 1을 그리시오", outputs["선분 1을 그리시오"])
        assigned_by_turtle = {}
        for turtle_id, _instruction, _started_at in log:
            assigned_by_turtle[turtle_id] = assigned_by_turtle.get(turtle_id, 0) + 1
        self.assertGreater(assigned_by_turtle["turtle1"], assigned_by_turtle["turtle2"])
        self.assertEqual(
            sorted(result.metadata["prompt_index"] for result in results),
            list(range(5)),
        )
        joined_logs = "\n".join(output_lines)
        self.assertIn("invoking worker LLM prompt#0", joined_logs)
        self.assertIn("completed worker task prompt#0", joined_logs)
        self.assertIn("\033[36m[turtle1]\033[0m", joined_logs)
        self.assertIn("\033[35m[turtle2]\033[0m", joined_logs)
        self.assertIn("worker_system_prompt", results[0].metadata)

    def test_prompt_queue_failure_does_not_stop_other_agents(self):
        log = []
        lock = threading.Lock()
        control = TurtleControlAgent(
            {
                "turtle1": _make_recording_worker(log, lock, fail_first=True),
                "turtle2": _make_recording_worker(log, lock),
            }
        )
        prompts = _plan_segment_prompts("세 개의 선분을 그리시오", count=3)

        results = control.run_prompt_queue(prompts)

        failures = [result for result in results if not result.ok]
        successes = [result for result in results if result.ok]
        self.assertEqual(len(failures), 1)
        self.assertIn("RuntimeError", failures[0].error)
        self.assertEqual(len(successes), 2)
        self.assertTrue(all("turtle2:" in result.output for result in successes))
        self.assertEqual(
            sorted(result.metadata["prompt_index"] for result in results),
            [0, 1, 2],
        )

    def test_user_prompt_calls_llm_planner_then_runs_prompt_queue(self):
        log = []
        output_lines = []
        lock = threading.Lock()
        planner = FakeLlmPlanner(
            _plan_segment_prompts("삼각형을 선분으로 나누어 그리시오", count=3)
        )
        control = TurtleControlAgent(
            {
                "turtle1": _make_recording_worker(log, lock),
                "turtle2": _make_recording_worker(log, lock),
            },
            log_enabled=True,
            log_sink=output_lines.append,
        )

        results = control.run_user_prompt("삼각형을 선분으로 나누어 그리시오", planner)

        self.assertEqual(len(planner.calls), 1)
        self.assertIn("여러 worker agent를 조율하는 컨트롤 에이전트", planner.calls[0])
        self.assertIn("사용자 요청:\n삼각형을 선분으로 나누어 그리시오", planner.calls[0])
        self.assertEqual(len(results), 3)
        self.assertTrue(all(result.ok for result in results))
        outputs = {result.instruction: result.output for result in results}
        self.assertIn("received 선분 0을 그리시오", outputs["선분 0을 그리시오"])
        self.assertIn("received 선분 2을 그리시오", outputs["선분 2을 그리시오"])
        joined_logs = "\n".join(output_lines)
        self.assertIn("[control] planning worker prompts with LLM", joined_logs)
        self.assertIn("[control] planned 3 worker prompts", joined_logs)
        self.assertIn("[control] queued worker prompt#0", joined_logs)

    def test_user_prompt_reports_planner_failure(self):
        class FailingPlanner:
            def invoke(self, _user_prompt):
                raise RuntimeError("llm unavailable")

        control = TurtleControlAgent({"turtle1": lambda task: task.instruction})

        results = control.run_user_prompt("정사각형을 그리시오", FailingPlanner())

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].ok)
        self.assertEqual(results[0].turtle_id, "control")
        self.assertIn("planner failed: RuntimeError: llm unavailable", results[0].error)


if __name__ == "__main__":
    unittest.main()
