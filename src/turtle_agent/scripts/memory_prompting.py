from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class MemoryContextResult:
    context: str
    hits: int
    query_context: Dict[str, Any]
    selected_records: Tuple[Dict[str, Any], ...] = ()
    policy_tags: Tuple[str, ...] = ()
    policy_reasons: Tuple[str, ...] = ()


def infer_query_context(query: str) -> Dict[str, Any]:
    text = str(query or "").strip()
    # IME/오타: 선두 한글 자모 짧은 접두 + 공백 후 영문 명령인 경우 접두 제거 (예: "ㅇ draw ...")
    m_prefix = re.match(r"^([\u3131-\u318e]{1,3})\s+(.*)$", text)
    if m_prefix and re.match(r"^[A-Za-z]", m_prefix.group(2)):
        text = m_prefix.group(2).strip()
    lowered = text.lower()
    slots: Dict[str, str] = {}

    # English style: "from A to B"
    m_en = re.search(r"\bfrom\s+([A-Za-z0-9_-]+)\s+to\s+([A-Za-z0-9_-]+)\b", lowered)
    if m_en:
        slots["from"] = m_en.group(1).upper()
        slots["to"] = m_en.group(2).upper()

    # Korean style: "A에서 B로 가"
    m_ko = re.search(r"([A-Za-z0-9_-]+)\s*에서\s*([A-Za-z0-9_-]+)\s*로", text)
    if m_ko:
        slots["from"] = m_ko.group(1).upper()
        slots["to"] = m_ko.group(2).upper()

    task_family = "natural_language_query"
    # NOTE: 현재 retrieval 품질 검증 범위가 navigation 중심이라 navigate 분기를 명시적으로 유지한다.
    # intent_norm(task_family=navigate, from/to slots)과 키를 맞춰 recall 일관성을 확보하는 목적이며,
    # 추후 trace_shape/rotate 확장 시에는 task별 parser/score 전략을 분리할 계획이다.
    # 좌표 기반 이동/선분 (memory 매칭을 navigate와 맞춤 — long intent_norm 과 일치시키기 위함)
    if re.search(
        r"\b(?:draw\s+(?:a\s+)?line|line)\s+to\s+(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)",
        lowered,
    ):
        task_family = "navigate"
    elif re.search(
        r"\b(?:move\s+back\s+to|return\s+to)\s+(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)",
        lowered,
    ):
        task_family = "navigate"
    elif slots.get("from") and slots.get("to"):
        task_family = "navigate"
    elif any(
        token in lowered
        for token in (
            "go to",
            "move to",
            "move back to",
            "return to",
            "goto",
            "이동",
            "가줘",
            "가 ",
            "로 가",
        )
    ):
        task_family = "navigate"
    elif any(token in lowered for token in ("star", "pentagram", "별", "오각별")):
        task_family = "trace_shape"
    elif any(token in lowered for token in ("rotate", "turn", "회전")):
        task_family = "rotate"

    if task_family == "navigate" and slots.get("from") and slots.get("to"):
        experience_key = f"navigate|from:{slots['from']}|to:{slots['to']}"
    else:
        experience_key = task_family

    return {
        "task_family": task_family,
        "slots": slots,
        "experience_key": experience_key,
    }


def load_long_term_records(memory_root: Path, turtle_id: str) -> List[Dict[str, Any]]:
    long_dir = memory_root / "long_term"
    if not long_dir.exists():
        return []
    out: List[Dict[str, Any]] = []
    for path in sorted(long_dir.glob("long_sessionid_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if str(row.get("turtle_id", "")) != str(turtle_id):
                continue
            out.append(row)
    return out


def _record_context(row: Dict[str, Any]) -> Dict[str, Any]:
    payload = row.get("payload", {})
    operation = payload.get("operation", {})
    intent_norm = operation.get("intent_norm", {})
    task_family = str(intent_norm.get("task_family", ""))
    slots = intent_norm.get("slots", {})
    if not isinstance(slots, dict):
        slots = {}
    from_slot = str(slots.get("from", "")).upper() if slots.get("from") else ""
    to_slot = str(slots.get("to", "")).upper() if slots.get("to") else ""
    key = task_family
    if task_family == "navigate" and from_slot and to_slot:
        key = f"navigate|from:{from_slot}|to:{to_slot}"
    return {
        "task_family": task_family,
        "slots": slots,
        "experience_key": key,
    }


def _safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _quality_band(row: Dict[str, Any]) -> str:
    payload = row.get("payload", {})
    evidence = payload.get("evidence", {})
    outcome = payload.get("outcome", {})
    success = bool(outcome.get("success", False))
    success_rate = _safe_float(evidence.get("success_rate"))
    collisions = _safe_int(evidence.get("collision_enter_count"), 0)
    if success and (success_rate is None or success_rate >= 0.8) and collisions <= 1:
        return "high"
    if (not success) or collisions >= 3 or (success_rate is not None and success_rate < 0.4):
        return "low"
    return "mid"


def _slot_specificity(record_ctx: Dict[str, Any]) -> int:
    slots = record_ctx.get("slots", {})
    has_from = bool(str(slots.get("from", "")).strip())
    has_to = bool(str(slots.get("to", "")).strip())
    return int(has_from) + int(has_to)


def _is_intent_match(query_ctx: Dict[str, Any], record_ctx: Dict[str, Any]) -> bool:
    task_family = str(query_ctx.get("task_family", "")).strip()
    if not task_family:
        return False
    return task_family == str(record_ctx.get("task_family", "")).strip()


def _score_record(query_ctx: Dict[str, Any], record_ctx: Dict[str, Any], quality: str) -> int:
    score = 0
    if query_ctx["experience_key"] and query_ctx["experience_key"] == record_ctx["experience_key"]:
        score += 35
    if query_ctx["task_family"] and query_ctx["task_family"] == record_ctx["task_family"]:
        score += 50
    q_slots = query_ctx.get("slots", {})
    r_slots = record_ctx.get("slots", {})
    if q_slots.get("from") and str(r_slots.get("from", "")).upper() == q_slots.get("from"):
        score += 20
    if q_slots.get("to") and str(r_slots.get("to", "")).upper() == q_slots.get("to"):
        score += 20
    # NOTE: 피드백에 따라 "성공 사례 우선" 대신 "실패 사례 우선" retrieval 정책을 적용한다.
    # 당장은 task-agnostic 규칙으로 일반화하지 않고, navigation 실효성 검증을 우선한다.
    if quality == "low":
        score += 8
    elif quality == "high":
        score -= 8
    return score


def _dedupe_key(record_ctx: Dict[str, Any]) -> str:
    slots = record_ctx.get("slots", {})
    from_slot = str(slots.get("from", "")).upper().strip()
    to_slot = str(slots.get("to", "")).upper().strip()
    return f"{record_ctx.get('task_family', '')}|from:{from_slot}|to:{to_slot}"


def _record_sort_tuple(score: int, record_ctx: Dict[str, Any], row: Dict[str, Any]) -> Tuple[int, int, int, float, int]:
    payload = row.get("payload", {})
    evidence = payload.get("evidence", {})
    success_rate = _safe_float(evidence.get("success_rate")) or 0.0
    collisions = _safe_int(evidence.get("collision_enter_count"), 0)
    created_at = _safe_int(row.get("meta", {}).get("created_at_unix_ms"), 0)
    # 정렬 우선순위(내림차순): score > slot_specificity > 충돌 많음 > 성공률 낮음 > 최신성
    return (score, _slot_specificity(record_ctx), collisions, -success_rate, created_at)


def _has_obstacle_policy_evidence(row: Dict[str, Any]) -> bool:
    payload = row.get("payload", {})
    evidence = payload.get("evidence", {})
    lessons = payload.get("lessons")
    if not isinstance(evidence, dict):
        return False
    if not isinstance(lessons, list) or not any(
        isinstance(lesson, str) and lesson.strip() for lesson in lessons
    ):
        return False
    if _safe_int(evidence.get("collision_enter_count"), 0) > 0:
        return True
    for key in (
        "collision_obstacles",
        "collision_temporary_obstacles",
        "collision_obstacle_geometries",
        "collision_hotspots",
    ):
        value = evidence.get(key)
        if isinstance(value, list) and value:
            return True
    return False


def _policy_tags_for_records(
    selected_records: List[Dict[str, Any]],
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    tags: set[str] = set()
    reasons: List[str] = []
    for row in selected_records:
        if not _has_obstacle_policy_evidence(row):
            continue
        tags.add("obstacle_validation")
        payload = row.get("payload", {})
        operation = payload.get("operation", {}) if isinstance(payload, dict) else {}
        goal_text = str(operation.get("nl_goal", {}).get("text", "")).strip()
        if goal_text:
            reasons.append(f"selected memory has obstacle evidence for goal={goal_text[:80]}")
        else:
            reasons.append("selected memory has obstacle evidence and lessons")
    return tuple(sorted(tags)), tuple(reasons)


def build_memory_context_result(
    query: str, records: List[Dict[str, Any]], top_k: int = 3
) -> MemoryContextResult:
    query_ctx = infer_query_context(query)
    max_k = max(0, int(top_k))
    if max_k == 0:
        return MemoryContextResult("", 0, query_ctx)
    min_score = 45
    deduped: Dict[str, Tuple[Tuple[int, int, int, float, int], Dict[str, Any], str, Dict[str, Any], int]] = {}
    for row in records:
        record_ctx = _record_context(row)
        if not _is_intent_match(query_ctx, record_ctx):
            continue
        quality = _quality_band(row)
        score = _score_record(query_ctx, record_ctx, quality)
        if score < min_score:
            continue
        sort_tuple = _record_sort_tuple(score, record_ctx, row)
        key = _dedupe_key(record_ctx)
        prev = deduped.get(key)
        if prev is None or sort_tuple > prev[0]:
            deduped[key] = (sort_tuple, row, quality, record_ctx, score)
    if not deduped:
        return MemoryContextResult("", 0, query_ctx)
    ranked = sorted(deduped.values(), key=lambda item: item[0], reverse=True)
    selected = ranked[: min(max_k, len(ranked))]
    selected_records = [item[1] for item in selected]
    lines: List[str] = []
    do_lines: List[str] = []
    dont_lines: List[str] = []
    for idx, (_, row, quality, _, score) in enumerate(selected, start=1):
        payload = row.get("payload", {})
        op = payload.get("operation", {})
        goal_text = str(op.get("nl_goal", {}).get("text", ""))
        evidence = payload.get("evidence", {})
        collisions = _safe_int(evidence.get("collision_enter_count"), 0)
        success_rate = _safe_float(evidence.get("success_rate"))

        collision_obstacles = evidence.get("collision_obstacles") or []
        obstacle_ids = (
            ", ".join(str(x) for x in collision_obstacles[:3]) if collision_obstacles else "없음"
        )

        collision_hotspots = evidence.get("collision_hotspots") or []
        collision_obstacle_geometries = evidence.get("collision_obstacle_geometries") or []
        location_tail = ""
        if isinstance(collision_hotspots, list) and collision_hotspots:
            location_tail = f" / collision_hotspots={collision_hotspots[:2]}"
        elif isinstance(collision_obstacle_geometries, list) and collision_obstacle_geometries:
            location_tail = f" / collision_obstacle_geometries={collision_obstacle_geometries[:2]}"

        goal_snippet = goal_text[:60]
        lines.append(
            f"{idx}. goal={goal_snippet} / collision_enter_count={collisions} / collision_obstacles={obstacle_ids} / success_rate={success_rate}{location_tail}"
        )
        raw_lessons = payload.get("lessons")
        if isinstance(raw_lessons, list):
            kept_lessons: List[str] = []
            for lesson in raw_lessons:
                if not (isinstance(lesson, str) and lesson.strip()):
                    continue
                # (즉시 적용 안전장치) 이미 저장된 lessons에 남아있는
                # 도구 성공/호출 수 관련 문구를 프롬프트에 섞지 않도록 제거합니다.
                forbidden_substrings = (
                    "모든 도구 단계가 성공적으로 끝났습니다",
                    "성공적으로 완료",
                    "도구 호출이 있었으며",
                    "개의 도구 호출",
                    "총 ",
                )
                if any(fs in lesson for fs in forbidden_substrings):
                    continue
                kept_lessons.append(lesson.strip())

            # (요구 반영) evidence의 장애물 geometry가 있으면,
            # lessons에서 위치 문장이 빠져도 합성해서 넣습니다.
            if isinstance(collision_obstacle_geometries, list) and collision_obstacle_geometries:
                geom0 = str(collision_obstacle_geometries[0])
                loc_sentence = f"충돌이 발생한 장애물 위치는 {geom0} 부근입니다."
                if not any(
                    ("장애물 위치" in lesson_text) or (geom0 in lesson_text)
                    for lesson_text in kept_lessons
                ):
                    kept_lessons.insert(0, loc_sentence)

            for lesson in kept_lessons:
                if quality == "low":
                    dont_lines.append(f"[memory {idx}] {lesson}")
                else:
                    do_lines.append(f"[memory {idx}] {lesson}")
    policy_lines = [
        "Use selected memory as execution evidence for this query.",
        "Do not introduce strategies that are absent from selected memory.",
    ]
    if dont_lines:
        policy_lines.append(
            "Treat DON'T items as constraints only when their stated conditions match."
        )
    policy_lines = [
        line for line in policy_lines if line.strip()
    ]
    context = "Memory policy (strict):\n"
    context += "\n".join(f"- {line}" for line in policy_lines)
    context += "\n\nMemory evidence:\n"
    context += "\n".join(lines)
    if do_lines:
        context += "\n\nDO rules:\n" + "\n".join(f"- {line}" for line in do_lines[:6])
    if dont_lines:
        context += "\n\nDON'T rules:\n" + "\n".join(f"- {line}" for line in dont_lines[:6])
    policy_tags, policy_reasons = _policy_tags_for_records(selected_records)
    return MemoryContextResult(
        context=context,
        hits=len(selected),
        query_context=query_ctx,
        selected_records=tuple(selected_records),
        policy_tags=policy_tags,
        policy_reasons=policy_reasons,
    )


def build_memory_context(query: str, records: List[Dict[str, Any]], top_k: int = 3) -> Tuple[str, int]:
    result = build_memory_context_result(query, records, top_k=top_k)
    return result.context, result.hits
