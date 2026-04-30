from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple


POINT_ALIAS = {
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "D",
    "에이": "A",
    "비": "B",
    "씨": "C",
    "디": "D",
    "에이점": "A",
    "비점": "B",
    "씨점": "C",
    "디점": "D",
    "A점": "A",
    "B점": "B",
    "C점": "C",
    "D점": "D",
}


def _has_korean(text: str) -> bool:
    return bool(re.search(r"[가-힣ㄱ-ㅎㅏ-ㅣ]", text))


def _normalize_point(raw: str) -> Optional[str]:
    token = str(raw).strip()
    if not token:
        return None
    token_upper = token.upper()
    if token_upper in ("A", "B", "C", "D"):
        return token_upper
    return POINT_ALIAS.get(token)


def _extract_route_slots(query: str) -> Dict[str, Any]:
    slots: Dict[str, Any] = {}
    m_ko = re.search(r"([A-Za-z가-힣]+)\s*에서\s*([A-Za-z가-힣]+)\s*로", query)
    if m_ko:
        src = _normalize_point(m_ko.group(1))
        dst = _normalize_point(m_ko.group(2))
        if src and dst:
            slots["from"] = src
            slots["to"] = dst
    m_xy = re.search(
        r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*에서\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*로",
        query,
    )
    if m_xy:
        slots["from_xy"] = (float(m_xy.group(1)), float(m_xy.group(2)))
        slots["to_xy"] = (float(m_xy.group(3)), float(m_xy.group(4)))
    m_turtle = re.search(r"(turtle\d+)", query, re.IGNORECASE)
    if m_turtle:
        slots["turtle"] = m_turtle.group(1).lower()
    return slots


def preprocess_korean_query(query: str) -> Dict[str, Any]:
    text = str(query or "").strip()
    lowered = text.lower()
    uses_korean = _has_korean(text)
    route_slots = _extract_route_slots(text)

    intent = "generic"
    confidence = 0.0
    allowed_tools = []

    if lowered in ("reset", "리셋", "초기화"):
        intent = "reset"
        confidence = 1.0
    elif any(k in lowered for k in ("이동", "가줘", "가 ", "경로", "회피", "에서")):
        intent = "navigate"
        confidence = 0.85 if route_slots else 0.65
        allowed_tools = [
            "list_obstacles",
            "check_path_against_obstacles",
            "draw_line_segment",
        ]

    hint_lines = []
    if uses_korean and intent == "navigate":
        hint_lines.append(
            "Korean preprocessing hint: This is a navigation request. Resolve route first, then call tools."
        )
        hint_lines.append(
            "Tool policy hint: If path can intersect obstacles, call list_obstacles/check_path_against_obstacles before draw_line_segment."
        )
        hint_lines.append(
            "Safety hint: Prefer segmented line drawing over long one-shot direct movement."
        )
        if route_slots.get("from") and route_slots.get("to"):
            hint_lines.append(
                f"Route slots: from={route_slots['from']} to={route_slots['to']}"
            )
        if route_slots.get("from_xy") and route_slots.get("to_xy"):
            hint_lines.append(
                f"Route slots: from_xy={route_slots['from_xy']} to_xy={route_slots['to_xy']}"
            )
        if route_slots.get("turtle"):
            hint_lines.append(f"Turtle slot: {route_slots['turtle']}")

    preprocessing_block = ""
    if hint_lines:
        preprocessing_block = "Preprocessing hints:\n" + "\n".join(
            f"- {line}" for line in hint_lines
        )

    return {
        "intent": intent,
        "confidence": confidence,
        "slots": route_slots,
        "allowed_tools": allowed_tools,
        "normalized_query": text,
        "preprocessing_block": preprocessing_block,
        "uses_korean": uses_korean,
    }

