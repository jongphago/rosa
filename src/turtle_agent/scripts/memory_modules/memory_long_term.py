from __future__ import annotations

"""Long-term compressed memory schema and pipeline.

Schema overview:
- record_id, record_type, turtle_id: 장기 레코드 식별 정보
- payload.operation: 자연어 목표/의도 정규화
- payload.action_trace: 실행 스텝의 직렬화된 시퀀스
- payload.outcome: 세션 성공 여부와 종료 사유
- payload.routine: 반복 가능한 스킬 시퀀스/파라미터 정책
- payload.evidence: 에피소드 수, 성공률, 충돌 집계
- payload.lessons: 세션 교훈 3문장
- meta: session_id, 압축 방식, 생성 시각
"""

import json
import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

_LESSONS_TARGET_LINES = 3
_LOGGER = logging.getLogger(__name__)


def _parse_lesson_lines(text: str, *, max_lines: int = _LESSONS_TARGET_LINES) -> List[str]:
    lines: List[str] = []
    for raw in (text or "").strip().splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^[\s>*-]*\d+[\).\s]+", "", line)
        line = line.lstrip("-•* ").strip()
        if line.startswith('"') and line.endswith('"') and len(line) > 2:
            line = line[1:-1].strip()
        if line:
            lines.append(line)
        if len(lines) >= max_lines:
            break
    return lines


def is_bootstrap_query_record(short_row: Dict[str, Any]) -> bool:
    goal = str(short_goal_text(short_row)).strip().lower()
    if not goal:
        return False
    if not any(token in goal for token in ("go to", "move to", "goto", "teleport")):
        return False
    return bool(re.search(r"\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?", goal))


def select_compression_batch(short_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(short_rows) >= 6 and is_bootstrap_query_record(short_rows[0]):
        return short_rows[1:]
    return short_rows


def infer_task_family(query: str) -> str:
    q = (query or "").lower()
    if "pentagram" in q or "star" in q:
        return "trace_shape"
    if "rotate" in q or "turn" in q:
        return "rotate"
    if any(token in q for token in ("go to", "move to", "goto", "teleport", "이동", "가줘", "로 가")):
        return "goto"
    return "natural_language_query"


def steps_equivalent(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if int(a.get("t_ms", 0)) != int(b.get("t_ms", 0)):
        return False
    ai = (a.get("skill_invocations") or [{}])[0]
    bi = (b.get("skill_invocations") or [{}])[0]
    return str(ai.get("skill", "")) == str(bi.get("skill", ""))


def short_goal_text(short_row: Dict[str, Any]) -> str:
    goal = short_row.get("goal", {})
    return str(goal.get("raw_text", "")) if isinstance(goal, dict) else ""


def short_trace_steps(short_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    ev = short_row.get("evidence", {})
    execution_steps = ev.get("execution_steps", []) if isinstance(ev, dict) else []
    if not isinstance(execution_steps, list):
        return out
    for item in execution_steps:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "t_ms": int(item.get("t_ms", 0)),
                "skill_invocations": [
                    {
                        "skill": item.get("skill", ""),
                        "args": item.get("args", {}),
                        "status": item.get("status", "unknown"),
                        "result": item.get("result", ""),
                    }
                ],
            }
        )
    return out


def short_collision_events(short_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    ev = short_row.get("evidence", {})
    events = ev.get("collision_events", []) if isinstance(ev, dict) else []
    return [e for e in events if isinstance(e, dict)] if isinstance(events, list) else []


def short_unix_ms(short_row: Dict[str, Any]) -> int:
    ds = short_row.get("decision_state", {})
    return int(ds.get("finalized_at_unix_ms", 0)) if isinstance(ds, dict) else 0


def is_strict_prefix_steps(prev_steps: List[Dict[str, Any]], curr_steps: List[Dict[str, Any]]) -> bool:
    if len(curr_steps) < len(prev_steps):
        return False
    for i in range(len(prev_steps)):
        if not steps_equivalent(prev_steps[i], curr_steps[i]):
            return False
    return True


def append_action_trace_from_short_row(
    action_trace: List[Dict[str, Any]],
    prev_steps: Optional[List[Dict[str, Any]]],
    curr_steps: List[Dict[str, Any]],
) -> None:
    def emit_step(step: Dict[str, Any]) -> None:
        for inv in step.get("skill_invocations") or []:
            action_trace.append(
                {
                    "t_ms": int(step.get("t_ms", 0)),
                    "skill": str(inv.get("skill", "")),
                    "args": inv.get("args", {}),
                    "status": str(inv.get("status", "unknown")),
                    "result": str(inv.get("result", "")),
                }
            )

    if not curr_steps:
        return
    if prev_steps is None:
        for step in curr_steps:
            emit_step(step)
        return
    if is_strict_prefix_steps(prev_steps, curr_steps):
        delta = curr_steps[len(prev_steps) :]
        if delta:
            for step in delta:
                emit_step(step)
        else:
            emit_step(curr_steps[-1])
        return
    for step in curr_steps:
        emit_step(step)


def extract_radius_series(action_trace: List[Dict[str, Any]]) -> List[float]:
    radii: List[float] = []
    for item in action_trace:
        result = str(item.get("result", ""))
        m = re.search(r"radius\s*\*{0,2}\s*([0-9]+(?:\.[0-9]+)?)", result, flags=re.I)
        if not m:
            continue
        try:
            radii.append(float(m.group(1)))
        except ValueError:
            continue
    return radii


def extract_context_from_action_trace(
    action_trace: List[Dict[str, Any]], fallback_goal: str
) -> Tuple[str, Dict[str, str]]:
    for item in action_trace:
        args = item.get("args", {})
        if not isinstance(args, dict):
            continue
        exp_key = str(args.get("experience_key", "")).strip()
        if exp_key.startswith("navigate|from:") and "|to:" in exp_key:
            m = re.match(r"^navigate\|from:([^|]+)\|to:(.+)$", exp_key)
            if m:
                return "navigate", {"from": m.group(1), "to": m.group(2)}
        if exp_key in ("navigate", "rotate", "trace_shape", "natural_language_query"):
            return exp_key, {}
        query = str(args.get("query", "")).strip()
        if query:
            inferred = infer_task_family(query)
            if inferred == "goto":
                return "navigate", {}
            return inferred, {}
    inferred_from_goal = infer_task_family(fallback_goal)
    if inferred_from_goal == "goto":
        return "navigate", {}
    return inferred_from_goal, {}


def collision_event_fingerprint(event: Dict[str, Any]) -> Tuple[Any, ...]:
    tr = event.get("t_ros")
    if isinstance(tr, dict):
        tkey = (int(tr.get("secs", 0)), int(tr.get("nsecs", 0)))
    else:
        tkey = (None, None)
    return (
        tkey,
        str(event.get("obstacle_id") or event.get("obstacle") or ""),
        str(event.get("event_type") or event.get("phase") or ""),
        str(event.get("collision_type") or ""),
    )


def collect_collision_evidence(short_term_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    collision_events = 0
    collision_enter_count = 0
    collision_obstacles: set[str] = set()
    collision_temporary_obstacles: set[str] = set()
    seen_fp: set[Tuple[Any, ...]] = set()
    for short in short_term_batch:
        for event in short_collision_events(short):
            row_kind = str(event.get("type", "")).lower()
            phase = str(event.get("phase") or event.get("event") or "").lower()
            subtype = str(event.get("event_type") or "").lower()
            collision_type = str(event.get("collision_type") or "").lower()
            if (
                "collision" not in row_kind
                and "collision" not in phase
                and "collision" not in collision_type
                and collision_type not in ("turtle_obstacle", "turtle_turtle")
            ):
                continue
            fp = collision_event_fingerprint(event)
            if fp in seen_fp:
                continue
            seen_fp.add(fp)
            collision_events += 1
            if phase in ("enter", "collision_enter") or subtype == "enter":
                collision_enter_count += 1
            obstacle = (
                event.get("obstacle")
                or event.get("obstacle_id")
                or event.get("name")
                or event.get("obstacle_name")
            )
            # a/b/c/d-point 같은 경유 지점 마커는 long-term 충돌 evidence에서 제외한다.
            if obstacle and str(obstacle).strip().lower().endswith("-point"):
                continue
            if obstacle:
                obstacle_name = str(obstacle)
                collision_obstacles.add(obstacle_name)
                details = event.get("details", {})
                details_kind = (
                    str(details.get("obstacle_kind", "")).lower()
                    if isinstance(details, dict)
                    else ""
                )
                obstacle_kind = str(event.get("obstacle_kind", "")).lower()
                if details_kind == "temporary" or obstacle_kind == "temporary":
                    collision_temporary_obstacles.add(obstacle_name)
    return {
        "collision_events": collision_events,
        "collision_enter_count": collision_enter_count,
        "collision_obstacles": sorted(collision_obstacles),
        "collision_temporary_obstacles": sorted(collision_temporary_obstacles),
    }


def lessons_context_payload(
    short_term_batch: List[Dict[str, Any]],
    *,
    collision_ev: Dict[str, Any],
    collision_obstacle_geometries: List[str],
    task_family: str,
    action_trace: List[Dict[str, Any]],
    first_goal: str,
    queries: List[str],
) -> Dict[str, Any]:
    # lessons 생성에는 "핵심 충돌 요약"만 필요하므로 입력 컨텍스트를 최소화한다.
    goal_primary = str(first_goal or "").strip()
    goal_latest = ""
    for q in queries:
        text = str(q or "").strip()
        if text:
            goal_latest = text
    if goal_latest == goal_primary:
        goal_latest = ""

    temporary_ids = collision_ev.get("collision_temporary_obstacles") or []
    if not isinstance(temporary_ids, list):
        temporary_ids = []
    temporary_ids = [str(x).strip() for x in temporary_ids if str(x).strip()][:5]

    # geometry는 문자열이 길어지기 쉬워 개수와 길이를 함께 제한한다.
    geometry_brief: List[str] = []
    for item in collision_obstacle_geometries[:3]:
        text = str(item or "").strip()
        if not text:
            continue
        geometry_brief.append(text[:120])

    return {
        "task_family": task_family,
        "goal_primary": goal_primary,
        "goal_latest": goal_latest,
        "collision_enter_count": int(collision_ev.get("collision_enter_count", 0)),
        "collision_temporary_obstacles": temporary_ids,
        # 장애물 geometry/좌표 요약을 lessons 생성에 함께 제공(길이 제한)
        "collision_obstacle_geometries": geometry_brief,
    }


def fallback_lessons_lines(
    *,
    collision_ev: Dict[str, Any],
    collision_obstacle_geometries: List[str],
    task_family: str,
    first_goal: str,
    action_trace: List[Dict[str, Any]],
) -> List[str]:
    enters = int(collision_ev.get("collision_enter_count", 0))
    temporary_obstacles = collision_ev.get("collision_temporary_obstacles") or []
    obstacles = temporary_obstacles or collision_ev.get("collision_obstacles") or []
    obs_txt = ", ".join(str(x) for x in obstacles[:5]) if obstacles else "없음"
    line1 = (
        f"이번 세션의 주요 목표는 「{first_goal[:120]}」이며 "
        f"작업 유형은 {task_family}로 분류되었습니다."
    )
    if enters > 0:
        line2 = (
            f"장애물 구간에서 충돌 진입이 {enters}회 기록되었고 "
            f"관련 장애물 식별자는 {obs_txt}입니다."
        )
    else:
        line2 = "기록된 충돌 진입은 없었고, 주행 중 장애물 관통 이벤트도 집계되지 않았습니다."
    # (중요) 도구 성공 여부/호출 수 같은 문구는 lessons에서 제거하고,
    # 장애물 위치(geometry) 기반으로 다음 행동 조심점을 더 직접적으로 전달합니다.
    geom_tail = ""
    if collision_obstacle_geometries:
        geom_tail = f"충돌이 발생한 장애물 위치는 {collision_obstacle_geometries[0]} 부근입니다."
    else:
        geom_tail = "충돌이 발생한 장애물 위치는 식별은 되었지만 좌표/형상 정보가 제한적입니다."
    line3 = geom_tail
    return [line1, line2, line3]


def summarize_lessons_with_llm(
    short_term_batch: List[Dict[str, Any]],
    *,
    collision_ev: Dict[str, Any],
    collision_obstacle_geometries: List[str],
    task_family: str,
    action_trace: List[Dict[str, Any]],
    first_goal: str,
    queries: List[str],
) -> List[str]:
    fb = fallback_lessons_lines(
        collision_ev=collision_ev,
        collision_obstacle_geometries=collision_obstacle_geometries,
        task_family=task_family,
        first_goal=first_goal or "(미상)",
        action_trace=action_trace,
    )
    if os.getenv("MEMORY_LESSONS_LLM", "1").strip().lower() in ("0", "false", "no", "off"):
        return fb

    payload = lessons_context_payload(
        short_term_batch,
        collision_ev=collision_ev,
        collision_obstacle_geometries=collision_obstacle_geometries,
        task_family=task_family,
        action_trace=action_trace,
        first_goal=first_goal,
        queries=queries,
    )
    try:
        from langchain_core.messages import HumanMessage

        from llm import get_llm

        ctx = json.dumps(payload, ensure_ascii=False, indent=2)
        if len(ctx) > 12000:
            ctx = ctx[:12000] + "\n…(truncated)"

        prompt = (
            "당신은 turtle_agent의 단기 메모리(short-term) 요약을 읽고 "
            "같은 세션에서 다음에 활용할 교훈만 추립니다.\n\n"
            "규칙:\n"
            "- 단기 기록에서 실제로 나타난 목표·행동·충돌·(장애물 geometry)만 근거로 씁니다. 추측은 최소화합니다.\n"
            "- 출력은 반드시 3문장 구성으로 하며, 각 문장은 한 줄에 하나씩입니다.\n"
            "- 1번째 문장: 충돌이 발생한 장애물 위치(geometry 요약)를 명시합니다.\n"
            "- 2번째 문장: 충돌 진입 횟수와 temporary 유형 장애물 식별자를 명시합니다.\n"
            "- 3번째 문장: 다음 실행에서 그 위치/상황을 피하거나 더 짧게 분절해 재계획하는 조심점을 1개 제시합니다.\n"
            "- 반드시 금지: '모든 도구 단계가 성공적으로 끝났습니다', '성공적으로 완료', '도구 호출이 있었으며', '총 N개의 도구' 같은 문구를 포함하지 마세요.\n"
            "- 반드시 금지: tool step 성공/실패(예: all_success) 전반을 설명하려는 문장을 쓰지 마세요.\n"
            "- 정확히 세 문장만 출력합니다. (다른 부가 문장/라벨 금지)\n"
            "- 번호, 글머리표, 따옴표 장식 없이 평문만 사용합니다.\n"
            "- 한국어로 작성합니다.\n\n"
            "입력 요약(JSON):\n"
            f"{ctx}"
        )
        llm = get_llm(streaming=False)
        msg = llm.invoke([HumanMessage(content=prompt)])
        raw = getattr(msg, "content", None)
        text = raw if isinstance(raw, str) else str(raw or "")
        parsed = _parse_lesson_lines(text, max_lines=_LESSONS_TARGET_LINES)
        out: List[str] = []
        for i in range(_LESSONS_TARGET_LINES):
            out.append(parsed[i] if i < len(parsed) else fb[i])
        return out
    except Exception as exc:
        _LOGGER.warning("long-term lessons LLM unavailable, using fallback: %s", exc)
        return fb


def create_long_term_record(
    short_term_batch: List[Dict[str, Any]],
    session_id: str,
    turtle_id: str,
    obstacle_store: Optional[Any] = None,
) -> Dict[str, Any]:
    first_goal = short_goal_text(short_term_batch[0])
    queries = [str(short_goal_text(short)).strip() for short in short_term_batch if str(short_goal_text(short)).strip()]
    action_trace: List[Dict[str, Any]] = []
    prev_trace_steps: Optional[List[Dict[str, Any]]] = None
    for short in short_term_batch:
        steps = short_trace_steps(short)
        if not steps:
            continue
        append_action_trace_from_short_row(action_trace, prev_trace_steps, steps)
        prev_trace_steps = steps

    all_success = all(item.get("status") == "success" for item in action_trace) if action_trace else False
    radii = extract_radius_series(action_trace)
    decay_ratio = 1.0
    if len(radii) >= 2 and radii[-2] > 0:
        decay_ratio = round(radii[-1] / radii[-2], 3)
    primary_skill = action_trace[0].get("skill", "unknown") if action_trace else "unknown"
    skill_sequence = [str(item.get("skill", "")) for item in action_trace if item.get("skill")]
    task_family, slots = extract_context_from_action_trace(action_trace, first_goal)
    collision_ev = collect_collision_evidence(short_term_batch)

    def _geometry_to_compact_string(geometry: Any) -> str:
        """
        obstacle_store geometry를 프롬프트에 넣기 좋은 짧은 문자열로 축약합니다.

        - circle: circle(cx, cy, r)
        - aabb: aabb(min_x, min_y, max_x, max_y)
        - segments: segments(n=..., bbox=(...))
        """
        if geometry is None:
            return "unknown_geometry"

        if hasattr(geometry, "cx") and hasattr(geometry, "cy") and hasattr(geometry, "r"):
            try:
                return (
                    f"circle(cx={float(geometry.cx):.2f},"
                    f"cy={float(geometry.cy):.2f},"
                    f"r={float(geometry.r):.2f})"
                )
            except Exception:
                return f"circle(type={type(geometry).__name__})"

        if (
            hasattr(geometry, "min_x")
            and hasattr(geometry, "min_y")
            and hasattr(geometry, "max_x")
            and hasattr(geometry, "max_y")
        ):
            try:
                return (
                    f"aabb(min_x={float(geometry.min_x):.2f},"
                    f"min_y={float(geometry.min_y):.2f},"
                    f"max_x={float(geometry.max_x):.2f},"
                    f"max_y={float(geometry.max_y):.2f})"
                )
            except Exception:
                return f"aabb(type={type(geometry).__name__})"

        if hasattr(geometry, "segments"):
            try:
                segments = list(getattr(geometry, "segments") or [])
                n = len(segments)
                xs: List[float] = []
                ys: List[float] = []
                for seg in segments:
                    (x1, y1), (x2, y2) = seg
                    xs.extend([float(x1), float(x2)])
                    ys.extend([float(y1), float(y2)])
                if xs and ys:
                    bbox = (
                        f"bbox=({min(xs):.2f},{min(ys):.2f})-({max(xs):.2f},{max(ys):.2f})"
                    )
                else:
                    bbox = "bbox=unknown"
                return f"segments(n={n},{bbox})"
            except Exception:
                return f"segments(type={type(geometry).__name__})"

        return f"unknown_geometry(type={type(geometry).__name__})"

    collision_obstacles = collision_ev.get("collision_obstacles") or []
    collision_obstacle_geometries: List[str] = []
    if obstacle_store is not None and collision_obstacles:
        # Geometry는 프롬프트 길이를 위해 일부만 제공합니다.
        for oid in collision_obstacles[:4]:
            try:
                ob = obstacle_store.get(oid)
            except Exception:
                ob = None
            if ob is None:
                continue
            geometry = getattr(ob, "geometry", None)
            geom_str = _geometry_to_compact_string(geometry)
            collision_obstacle_geometries.append(f"{oid}:{geom_str}")
    lessons_lines = summarize_lessons_with_llm(
        short_term_batch,
        collision_ev=collision_ev,
        collision_obstacle_geometries=collision_obstacle_geometries,
        task_family=task_family,
        action_trace=action_trace,
        first_goal=str(first_goal or ""),
        queries=queries,
    )
    return {
        "record_id": str(uuid.uuid4()),
        "record_type": "compressed_routine",
        "turtle_id": turtle_id,
        "payload": {
            "operation": {
                "nl_goal": {
                    "text": f"{first_goal} (follow-ups: {' / '.join(queries[1:])})"
                    if len(queries) > 1
                    else first_goal
                },
                "intent_norm": {
                    "task_family": task_family,
                    "slots": {"shape": "pentagram"} if task_family == "trace_shape" else slots,
                },
            },
            "action_trace": action_trace,
            "outcome": {
                "success": all_success,
                "terminal_reason": "goal_reached" if all_success else "execution_failed",
            },
            "routine": {
                "name": "pentagram_repeat_with_scaling" if task_family == "trace_shape" else "query_routine",
                "skill_sequence": skill_sequence or ([primary_skill] if primary_skill != "unknown" else []),
                "default_args": {"center_x": 5.544, "center_y": 5.544} if task_family == "trace_shape" else {},
                "param_policy": {
                    "radius_decay_ratio": decay_ratio,
                    "followup_rules": [
                        {"trigger": "another one", "action": "reuse previous shape"},
                        {"trigger": "smaller", "action": "apply radius decay"},
                    ],
                }
                if task_family == "trace_shape"
                else {},
            },
            "evidence": {
                "n_episodes": len(action_trace),
                "success_rate": round(
                    sum(1 for item in action_trace if item.get("status") == "success") / max(1, len(action_trace)),
                    3,
                ),
                "collision_events": collision_ev["collision_events"],
                "collision_enter_count": collision_ev["collision_enter_count"],
                "collision_obstacles": collision_ev["collision_obstacles"],
                "collision_temporary_obstacles": collision_ev["collision_temporary_obstacles"],
                # (B) now: obstacle geometry 요약으로 제공.
                # (A) later: collision_hotspots 필드를 추가로 채워도(또는 값만 교체해도) 프롬프트는 둘 다 확인하도록 구성할 예정입니다.
                "collision_obstacle_geometries": collision_obstacle_geometries,
                "collision_hotspots": [],
            },
            "lessons": lessons_lines,
        },
        "meta": {
            "session_id": session_id,
            "compression": {"method": "deterministic"},
            "created_at_unix_ms": short_unix_ms(short_term_batch[-1]),
        },
    }

