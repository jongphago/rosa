#!/usr/bin/env python3
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

"""Runtime for running the control agent against a live turtlesim instance."""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import rospy
import tools.turtle as turtle_tools
from geometry_msgs.msg import Twist
from llm import get_llm
from obstacle_store import ObstacleStore
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from ros_params import get_bool_param
from turtle_control_agent import TurtleControlAgent, TurtleTask, TurtleTaskResult
from turtle_control_prompts import WORKER_SYSTEM_PROMPT

try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain.prompts import MessagesPlaceholder
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:
    AgentExecutor = None
    ChatPromptTemplate = None
    MessagesPlaceholder = None
    create_tool_calling_agent = None


_DEFAULT_PROMPT_LIMIT = 0
_DEFAULT_TIMEOUT = 0.0


class LangChainPlanner:
    """Planner that turns a user request into line-delimited worker tasks."""

    def __init__(self, *, task_count: int = _DEFAULT_PROMPT_LIMIT, streaming: bool = False) -> None:
        self.task_count = task_count
        self.llm = get_llm(streaming=streaming)

    def invoke(self, control_prompt: str) -> str:
        planner_prompt = control_prompt
        if self.task_count > 0:
            planner_prompt = (
                f"{control_prompt}\n\n"
                f"생성할 worker 작업 수는 최대 {self.task_count}개입니다."
            )
        response = self.llm.invoke(planner_prompt)
        return _message_to_text(response)


class TurtleSimWorker:
    """LLM worker that operates one assigned turtle through turtlesim tools."""

    def __init__(self, turtle_name: str, *, streaming: bool = False) -> None:
        self.turtle_name = turtle_name
        self.llm = get_llm(streaming=streaming)
        self.tools = _turtlesim_tools()
        self.executor = _build_worker_executor(
            llm=self.llm,
            tools=self.tools,
            streaming=streaming,
        )

    def __call__(self, task: TurtleTask) -> str:
        result = self.executor.invoke(
            {
                "worker_system_prompt": task.metadata.get(
                    "worker_system_prompt",
                    WORKER_SYSTEM_PROMPT,
                ),
                "turtle_name": self.turtle_name,
                "input": task.instruction,
            }
        )
        if isinstance(result, dict) and "output" in result:
            return str(result["output"])
        return _message_to_text(result)


def run_turtle_control_agent(
    *,
    obstacle_store: Optional[ObstacleStore] = None,
    lifecycle_listener: Optional[Any] = None,
) -> None:
    """Run the interactive control-agent loop inside the initialized ROS node."""
    _ = obstacle_store
    console = Console()
    streaming = get_bool_param("~streaming", False)
    worker_count = _get_int_param("~control_worker_count", 2)
    task_count = _get_int_param("~control_task_count", _DEFAULT_PROMPT_LIMIT)
    timeout = _get_float_param("~control_task_timeout", _DEFAULT_TIMEOUT)
    reset_turtlesim = get_bool_param("~control_reset_turtlesim", False)

    turtle_names = _turtle_names(worker_count)
    planner = LangChainPlanner(task_count=task_count, streaming=streaming)
    control_agent = TurtleControlAgent({}, log_enabled=False)
    control_lifecycle = TurtleControlLifecycleListener(
        control_agent,
        worker_factory=lambda turtle_name: TurtleSimWorker(
            turtle_name,
            streaming=streaming,
        ),
    )
    turtle_tools.configure_turtle_lifecycle_listener(
        CompositeTurtleLifecycleListener(lifecycle_listener, control_lifecycle)
    )
    _prepare_turtles(turtle_names, reset_turtlesim=reset_turtlesim)
    for turtle_name in turtle_names:
        control_lifecycle.on_turtle_spawned(turtle_name)

    console.print(
        Panel(
            Markdown(
                "Turtle control mode is running.\n\n"
                "- Type a drawing request to split it into worker tasks.\n"
                "- Type `exit` to quit."
            ),
            title="ROSA Turtle Control",
            border_style="green",
        )
    )

    while not rospy.is_shutdown():
        try:
            user_prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[Shutdown complete]")
            break
        if not user_prompt:
            continue
        if user_prompt == "exit":
            break

        results = control_agent.run_user_prompt(
            user_prompt,
            planner,
            timeout=None if timeout <= 0 else timeout,
        )
        _print_results(console, results)


class TurtleControlLifecycleListener:
    """Keep a control-agent worker registry in sync with turtle lifecycle events."""

    def __init__(
        self,
        control_agent: TurtleControlAgent,
        *,
        worker_factory: Callable[[str], TurtleSimWorker],
    ) -> None:
        self._control_agent = control_agent
        self._worker_factory = worker_factory

    def on_turtle_spawned(self, name: str) -> None:
        turtle_name = _normalize_turtle_name(name)
        if not turtle_name:
            return
        if turtle_name in self._control_agent.worker_ids():
            turtle_tools.add_cmd_vel_pub(
                turtle_name,
                rospy.Publisher(f"/{turtle_name}/cmd_vel", Twist, queue_size=10),
            )
            return
        self._control_agent.add_worker(
            turtle_name,
            self._worker_factory(turtle_name),
        )
        turtle_tools.add_cmd_vel_pub(
            turtle_name,
            rospy.Publisher(f"/{turtle_name}/cmd_vel", Twist, queue_size=10),
        )

    def on_turtle_killed(self, name: str) -> None:
        turtle_name = _normalize_turtle_name(name)
        if not turtle_name:
            return
        self._control_agent.remove_worker(turtle_name)
        turtle_tools.remove_cmd_vel_pub(turtle_name)


class CompositeTurtleLifecycleListener:
    """Fan out turtle lifecycle notifications to multiple listeners."""

    def __init__(self, *listeners: Optional[Any]) -> None:
        self._listeners = tuple(listener for listener in listeners if listener is not None)

    def on_turtle_spawned(self, name: str) -> None:
        for listener in self._listeners:
            listener.on_turtle_spawned(name)

    def on_turtle_killed(self, name: str) -> None:
        for listener in self._listeners:
            listener.on_turtle_killed(name)


def _build_worker_executor(llm: Any, tools: Sequence[Any], streaming: bool) -> Any:
    if not _tool_calling_available():
        raise RuntimeError("LangChain tool-calling agent dependencies are unavailable.")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{worker_system_prompt}"),
            (
                "human",
                "assigned_turtle: {turtle_name}\n"
                "task: {input}\n\n"
                "Use assigned_turtle as the `name` argument for movement and drawing "
                "tools. Spawn and kill tools may use the turtle name requested by "
                "the task.",
            ),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(
        llm=llm.with_config({"streaming": streaming}),
        tools=tools,
        prompt=prompt,
    )
    return AgentExecutor(
        agent=agent,
        tools=tools,
        stream_runnable=streaming,
        verbose=False,
        max_iterations=20,
        handle_parsing_errors=True,
    )


def _tool_calling_available() -> bool:
    return all(
        item is not None
        for item in (
            AgentExecutor,
            ChatPromptTemplate,
            MessagesPlaceholder,
            create_tool_calling_agent,
        )
    )


def _turtlesim_tools() -> list[Any]:
    return [
        turtle_tools.spawn_turtle,
        turtle_tools.kill_turtle,
        turtle_tools.clear_turtlesim,
        turtle_tools.get_turtle_pose,
        turtle_tools.teleport_absolute,
        turtle_tools.teleport_relative,
        turtle_tools.publish_twist_to_cmd_vel,
        turtle_tools.stop_turtle,
        turtle_tools.set_pen,
        turtle_tools.has_moved_to_expected_coordinates,
        turtle_tools.draw_line_segment,
        turtle_tools.draw_rectangle,
        turtle_tools.draw_polyline,
        turtle_tools.calculate_rectangle_bounds,
        turtle_tools.check_rectangles_overlap,
        turtle_tools.draw_circle,
        turtle_tools.draw_arc,
    ]


def _prepare_turtles(turtle_names: Sequence[str], *, reset_turtlesim: bool) -> None:
    if reset_turtlesim:
        rospy.loginfo("%s", turtle_tools.reset_turtlesim.invoke({}))

    for index, turtle_name in enumerate(turtle_names):
        if turtle_name != "turtle1":
            x, y = _spawn_position(index)
            result = turtle_tools.spawn_turtle.invoke(
                {
                    "name": turtle_name,
                    "x": x,
                    "y": y,
                    "theta": 0.0,
                }
            )
            rospy.loginfo("%s", result)


def _spawn_position(index: int) -> tuple[float, float]:
    positions = (
        (5.5, 5.5),
        (2.0, 2.0),
        (9.0, 2.0),
        (2.0, 9.0),
        (9.0, 9.0),
    )
    return positions[index % len(positions)]


def _turtle_names(worker_count: int) -> tuple[str, ...]:
    names_param = str(rospy.get_param("~control_turtle_names", "")).strip()
    if names_param:
        names = tuple(name.strip().strip("/") for name in names_param.split(",") if name.strip())
        if names:
            return names
    return tuple(f"turtle{index}" for index in range(1, max(worker_count, 1) + 1))


def _normalize_turtle_name(name: str) -> str:
    return str(name).strip().replace("/", "")


def _get_int_param(name: str, default: int) -> int:
    try:
        return int(rospy.get_param(name, default))
    except (TypeError, ValueError):
        return default


def _get_float_param(name: str, default: float) -> float:
    try:
        return float(rospy.get_param(name, default))
    except (TypeError, ValueError):
        return default


def _print_results(console: Console, results: Sequence[TurtleTaskResult]) -> None:
    lines = []
    for result in results:
        status = "ok" if result.ok else "failed"
        detail = result.output if result.ok else result.error
        lines.append(f"- `{result.turtle_id}` {status}: {detail}")
    console.print(Panel(Markdown("\n".join(lines)), title="Worker Results"))


def _message_to_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                chunks.append(str(item.get("text", "")))
            else:
                chunks.append(str(item))
        return "\n".join(chunk for chunk in chunks if chunk.strip())
    return str(content)
