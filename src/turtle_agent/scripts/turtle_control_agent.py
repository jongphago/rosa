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

"""ROS-free orchestration for running one worker agent per turtle in parallel."""

from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    TimeoutError,
    as_completed,
    wait,
)
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

from turtle_control_prompts import CONTROL_AGENT_PROMPT, WORKER_SYSTEM_PROMPT

_ANSI_RESET = "\033[0m"
_AGENT_COLORS = (
    "\033[36m",  # cyan
    "\033[35m",  # magenta
    "\033[33m",  # yellow
    "\033[32m",  # green
    "\033[34m",  # blue
    "\033[31m",  # red
)

DEFAULT_CONTROL_AGENT_PROMPT = CONTROL_AGENT_PROMPT
DEFAULT_WORKER_SYSTEM_PROMPT = WORKER_SYSTEM_PROMPT


@dataclass(frozen=True)
class TurtleTask:
    """A unit of work addressed to one turtle-specific worker agent."""

    turtle_id: str
    instruction: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TurtleTaskResult:
    """Result for one turtle task, including isolated failures."""

    turtle_id: str
    ok: bool
    instruction: str = ""
    output: str = ""
    error: str = ""
    elapsed_seconds: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


TurtleWorker = Callable[[TurtleTask], str]
LogSink = Callable[[str], None]
PromptPlanner = Callable[[str], Sequence[str]]


class TurtleControlAgent:
    """
    Coordinate turtle-specific worker agents without importing ROS or turtlesim.

    The control agent owns only orchestration: each worker is responsible for a
    single turtle, while this class runs independent turtle tasks concurrently
    and collects per-turtle results.
    """

    def __init__(
        self,
        workers: Mapping[str, TurtleWorker],
        *,
        max_workers: Optional[int] = None,
        log_enabled: bool = False,
        log_sink: Optional[LogSink] = None,
        control_prompt: str = DEFAULT_CONTROL_AGENT_PROMPT,
        worker_system_prompt: str = DEFAULT_WORKER_SYSTEM_PROMPT,
    ) -> None:
        self._worker_lock = threading.RLock()
        self._workers = dict(workers)
        if max_workers is not None and max_workers < 1:
            raise ValueError("max_workers must be positive when provided")
        self._max_workers = max_workers
        self._log_enabled = log_enabled
        self._log_sink = log_sink or print
        self._agent_colors = {
            worker_id: _AGENT_COLORS[index % len(_AGENT_COLORS)]
            for index, worker_id in enumerate(self._workers)
        }
        self._control_prompt = control_prompt
        self._worker_system_prompt = worker_system_prompt

    def add_worker(self, turtle_id: str, worker: TurtleWorker) -> None:
        """Register or replace the worker for ``turtle_id``."""
        if not turtle_id:
            raise ValueError("turtle_id must be non-empty")
        with self._worker_lock:
            self._workers[turtle_id] = worker
            if turtle_id not in self._agent_colors:
                color_index = len(self._agent_colors) % len(_AGENT_COLORS)
                self._agent_colors[turtle_id] = _AGENT_COLORS[color_index]

    def remove_worker(self, turtle_id: str) -> bool:
        """Remove the worker for ``turtle_id`` and return whether one existed."""
        with self._worker_lock:
            removed = self._workers.pop(turtle_id, None) is not None
            self._agent_colors.pop(turtle_id, None)
            return removed

    def worker_ids(self) -> Tuple[str, ...]:
        """Return currently registered worker ids."""
        with self._worker_lock:
            return tuple(self._workers)

    def run_parallel(
        self,
        tasks: Sequence[TurtleTask],
        *,
        timeout: Optional[float] = None,
    ) -> Tuple[TurtleTaskResult, ...]:
        """
        Run turtle tasks concurrently and return results in input order.

        Unknown turtles and worker exceptions become failed results instead of
        interrupting unrelated turtle tasks.
        """
        if not tasks:
            return tuple()

        workers = self._snapshot_workers()
        results: Dict[int, TurtleTaskResult] = {}
        future_context: Dict[Future, Tuple[int, TurtleTask, float]] = {}
        runnable_count = sum(1 for task in tasks if task.turtle_id in workers)
        pool_size = self._pool_size(runnable_count, worker_count=len(workers))

        if pool_size == 0:
            return tuple(
                self._unknown_turtle_result(task)
                for task in tasks
            )

        with ThreadPoolExecutor(max_workers=pool_size) as executor:
            for index, task in enumerate(tasks):
                worker = workers.get(task.turtle_id)
                if worker is None:
                    results[index] = self._unknown_turtle_result(task)
                    continue
                started_at = monotonic()
                self._log_start(task)
                future = executor.submit(worker, task)
                future_context[future] = (index, task, started_at)

            try:
                for future in as_completed(future_context, timeout=timeout):
                    index, task, started_at = future_context[future]
                    results[index] = self._result_from_future(future, task, started_at)
                    self._log_result(results[index])
            except TimeoutError:
                for future, (index, task, started_at) in future_context.items():
                    if future.done():
                        results[index] = self._result_from_future(future, task, started_at)
                    else:
                        future.cancel()
                        results[index] = TurtleTaskResult(
                            turtle_id=task.turtle_id,
                            ok=False,
                            instruction=task.instruction,
                            error="task timed out",
                            elapsed_seconds=monotonic() - started_at,
                            metadata=task.metadata,
                        )
                    self._log_result(results[index])

        return tuple(results[index] for index in range(len(tasks)))

    def run_prompt_queue(
        self,
        prompts: Sequence[str],
        *,
        timeout: Optional[float] = None,
    ) -> Tuple[TurtleTaskResult, ...]:
        """
        Feed prompts to worker agents until the shared queue is exhausted.

        A worker receives one prompt at a time. When it finishes successfully,
        the control agent immediately assigns the next queued prompt to that
        same worker. Failed workers are not reused, so other workers can keep
        draining the queue.
        """
        workers = self._snapshot_workers()
        if not prompts or not workers:
            return tuple()

        prompt_queue: Deque[Tuple[int, str]] = deque(enumerate(prompts))
        worker_ids = list(workers)
        pool_size = self._pool_size(
            min(len(worker_ids), len(prompt_queue)),
            worker_count=len(workers),
        )
        if pool_size == 0:
            return tuple()

        results: List[TurtleTaskResult] = []
        future_context: Dict[Future, Tuple[str, TurtleTask, float]] = {}
        deadline = None if timeout is None else monotonic() + timeout

        def _submit_next(executor: ThreadPoolExecutor, worker_id: str) -> None:
            prompt_index, prompt = prompt_queue.popleft()
            task = TurtleTask(
                turtle_id=worker_id,
                instruction=prompt,
                metadata={
                    "prompt_index": prompt_index,
                    "worker_system_prompt": self._worker_system_prompt,
                },
            )
            started_at = monotonic()
            self._log_start(task)
            future = executor.submit(workers[worker_id], task)
            future_context[future] = (worker_id, task, started_at)

        with ThreadPoolExecutor(max_workers=pool_size) as executor:
            for worker_id in worker_ids[:pool_size]:
                if not prompt_queue:
                    break
                _submit_next(executor, worker_id)

            while future_context:
                wait_timeout = None
                if deadline is not None:
                    wait_timeout = max(0.0, deadline - monotonic())
                    if wait_timeout == 0.0:
                        self._mark_active_as_timed_out(future_context, results)
                        break

                done, _pending = wait(
                    tuple(future_context),
                    timeout=wait_timeout,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    self._mark_active_as_timed_out(future_context, results)
                    break

                for future in done:
                    worker_id, task, started_at = future_context.pop(future)
                    result = self._result_from_future(future, task, started_at)
                    results.append(result)
                    self._log_result(result)
                    if result.ok and prompt_queue:
                        _submit_next(executor, worker_id)

        while prompt_queue:
            prompt_index, prompt = prompt_queue.popleft()
            results.append(
                TurtleTaskResult(
                    turtle_id="",
                    ok=False,
                    instruction=prompt,
                    error="no active worker available",
                    metadata={"prompt_index": prompt_index},
                )
            )

        return tuple(results)

    def run_user_prompt(
        self,
        user_prompt: str,
        llm_planner: Any,
        *,
        timeout: Optional[float] = None,
    ) -> Tuple[TurtleTaskResult, ...]:
        """
        Call an LLM/planner to turn one user prompt into worker prompts, then run them.

        `llm_planner` may be a callable (`planner(user_prompt)`) or a LangChain-like
        object with `invoke(user_prompt)`. Tests can pass a fake planner; production
        code can pass an actual LLM chain later.
        """
        self._log_llm_start(user_prompt)
        try:
            planner_prompt = self._build_control_prompt(user_prompt)
            planned = self._invoke_planner(llm_planner, planner_prompt)
            prompts = self._normalize_planned_prompts(planned)
        except Exception as exc:
            result = TurtleTaskResult(
                turtle_id="control",
                ok=False,
                instruction=user_prompt,
                error=f"planner failed: {type(exc).__name__}: {exc}",
            )
            self._log_result(result)
            return (result,)

        self._log_llm_result(user_prompt, prompts)
        return self.run_prompt_queue(prompts, timeout=timeout)

    def _build_control_prompt(self, user_prompt: str) -> str:
        return self._control_prompt.format(user_prompt=user_prompt)

    def _snapshot_workers(self) -> Dict[str, TurtleWorker]:
        with self._worker_lock:
            return dict(self._workers)

    def _pool_size(self, runnable_count: int, *, worker_count: Optional[int] = None) -> int:
        if runnable_count < 1:
            return 0
        if self._max_workers is not None:
            return min(self._max_workers, runnable_count)
        if worker_count is None:
            worker_count = len(self._snapshot_workers())
        return min(worker_count, runnable_count)

    def _agent_label(self, turtle_id: str) -> str:
        with self._worker_lock:
            color = self._agent_colors.get(turtle_id, "")
        label = f"[{turtle_id or 'unassigned'}]"
        if not color:
            return label
        return f"{color}{label}{_ANSI_RESET}"

    def _emit_log(self, message: str) -> None:
        if self._log_enabled:
            self._log_sink(message)

    def _log_llm_start(self, _user_prompt: str) -> None:
        self._emit_log("[control] planning worker prompts with LLM")

    def _log_llm_result(self, _user_prompt: str, prompts: Sequence[str]) -> None:
        self._emit_log(f"[control] planned {len(prompts)} worker prompts")
        for index, _prompt in enumerate(prompts):
            self._emit_log(f"[control] queued worker prompt#{index}")

    def _log_start(self, task: TurtleTask) -> None:
        task_ref = self._task_ref(task.metadata)
        self._emit_log(
            f"{self._agent_label(task.turtle_id)} invoking worker LLM {task_ref}"
        )

    def _log_result(self, result: TurtleTaskResult) -> None:
        task_ref = self._task_ref(result.metadata)
        if result.ok:
            self._emit_log(
                f"{self._agent_label(result.turtle_id)} completed worker task {task_ref}"
            )
            return
        self._emit_log(
            f"{self._agent_label(result.turtle_id)} worker task failed "
            f"{task_ref}: {result.error}"
        )

    @staticmethod
    def _task_ref(metadata: Dict[str, Any]) -> str:
        prompt_index = metadata.get("prompt_index")
        if prompt_index is None:
            return ""
        return f"prompt#{prompt_index}"

    @staticmethod
    def _invoke_planner(llm_planner: Any, user_prompt: str) -> Any:
        invoke = getattr(llm_planner, "invoke", None)
        if callable(invoke):
            return invoke(user_prompt)
        if callable(llm_planner):
            return llm_planner(user_prompt)
        raise TypeError("llm_planner must be callable or expose invoke(prompt)")

    @staticmethod
    def _normalize_planned_prompts(planned: Any) -> Tuple[str, ...]:
        if isinstance(planned, str):
            prompts = tuple(
                TurtleControlAgent._clean_planned_prompt(line)
                for line in planned.splitlines()
                if TurtleControlAgent._clean_planned_prompt(line)
            )
        else:
            prompts = tuple(
                TurtleControlAgent._clean_planned_prompt(str(prompt))
                for prompt in planned
                if TurtleControlAgent._clean_planned_prompt(str(prompt))
            )
        if not prompts:
            raise ValueError("planner returned no worker prompts")
        return prompts

    @staticmethod
    def _clean_planned_prompt(prompt: str) -> str:
        cleaned = prompt.strip()
        cleaned = cleaned.removeprefix("-").strip()
        cleaned = cleaned.removeprefix("*").strip()
        if "." in cleaned:
            prefix, rest = cleaned.split(".", 1)
            if prefix.strip().isdigit():
                cleaned = rest.strip()
        return cleaned

    @staticmethod
    def _unknown_turtle_result(task: TurtleTask) -> TurtleTaskResult:
        return TurtleTaskResult(
            turtle_id=task.turtle_id,
            ok=False,
            instruction=task.instruction,
            error=f"no worker registered for turtle '{task.turtle_id}'",
            metadata=task.metadata,
        )

    @staticmethod
    def _mark_active_as_timed_out(
        future_context: Dict[Future, Tuple[str, TurtleTask, float]],
        results: List[TurtleTaskResult],
    ) -> None:
        for future, (_worker_id, task, started_at) in tuple(future_context.items()):
            if future.done():
                results.append(
                    TurtleControlAgent._result_from_future(future, task, started_at)
                )
            else:
                future.cancel()
                results.append(
                    TurtleTaskResult(
                        turtle_id=task.turtle_id,
                        ok=False,
                        instruction=task.instruction,
                        error="task timed out",
                        elapsed_seconds=monotonic() - started_at,
                        metadata=task.metadata,
                    )
                )
            future_context.pop(future, None)

    @staticmethod
    def _result_from_future(
        future: Future,
        task: TurtleTask,
        started_at: float,
    ) -> TurtleTaskResult:
        elapsed = monotonic() - started_at
        try:
            output = future.result()
        except Exception as exc:
            return TurtleTaskResult(
                turtle_id=task.turtle_id,
                ok=False,
                instruction=task.instruction,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=elapsed,
                metadata=task.metadata,
            )
        return TurtleTaskResult(
            turtle_id=task.turtle_id,
            ok=True,
            instruction=task.instruction,
            output=str(output),
            elapsed_seconds=elapsed,
            metadata=task.metadata,
        )
