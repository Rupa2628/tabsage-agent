"""Guardrails: the checks that stop the agent from doing something wrong or getting stuck."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


class ParseError(Exception):
    """The model's reply could not be turned into a valid action."""


class UnsafeSQLError(Exception):
    """The SQL is not a single read-only query."""


@dataclass
class Action:
    thought: str
    name: str
    args: dict = field(default_factory=dict)


# ---------------------------------------------------------------- action parsing
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(text: str) -> dict:
    text = text.strip()
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    # also try the outermost {...} span, for replies with chatter around the JSON
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise ParseError("Reply was not a JSON object.")


def parse_action(text: str, allowed: set[str]) -> Action:
    obj = _extract_json(text)
    name = obj.get("action")
    if not isinstance(name, str) or not name:
        raise ParseError('JSON is missing the "action" field.')
    if name not in allowed:
        raise ParseError(f'Unknown action "{name}". Allowed: {sorted(allowed)}')
    args = obj.get("args", {})
    if not isinstance(args, dict):
        raise ParseError('"args" must be a JSON object.')
    return Action(thought=str(obj.get("thought", "")), name=name, args=args)


def parse_plan(text: str) -> list[str]:
    obj = _extract_json(text)
    plan = obj.get("plan")
    if not isinstance(plan, list) or not plan:
        raise ParseError('Plan JSON must contain a non-empty "plan" list.')
    return [str(s) for s in plan][:8]


# ---------------------------------------------------------------- SQL safety
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex)\b",
    re.IGNORECASE,
)


def validate_sql(query: str) -> str:
    """Allow exactly one SELECT / WITH statement. This is layer 1; the database is
    also opened read-only (layer 2), so a bypass here still cannot change data."""
    if not isinstance(query, str) or not query.strip():
        raise UnsafeSQLError("Query is empty.")
    q = query.strip().rstrip(";").strip()
    if ";" in q:
        raise UnsafeSQLError("Only one statement is allowed.")
    if not re.match(r"^(select|with)\b", q, re.IGNORECASE):
        raise UnsafeSQLError("Only SELECT (or WITH ... SELECT) queries are allowed.")
    # strip string literals before keyword scan so WHERE status = 'delete' is fine
    if _FORBIDDEN.search(re.sub(r"'(?:[^']|'')*'", "''", q)):
        raise UnsafeSQLError("Query contains a write or admin keyword.")
    return q


# ---------------------------------------------------------------- loop detection
class LoopDetector:
    """Flags an action the agent has already taken with identical arguments."""

    def __init__(self):
        self._seen: set[str] = set()

    @staticmethod
    def _key(action: Action) -> str:
        args = json.dumps(action.args, sort_keys=True)
        return f"{action.name}:{re.sub(r'\s+', ' ', args.lower())}"

    def is_repeat(self, action: Action) -> bool:
        k = self._key(action)
        if k in self._seen:
            return True
        self._seen.add(k)
        return False


# ---------------------------------------------------------------- answer grounding
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def numbers_in(text: str) -> list[float]:
    return [float(x) for x in _NUM.findall(text.replace(",", ""))]


def is_grounded(value: float, observations: list[str], rel_tol: float = 0.005) -> bool:
    """True if ``value`` (or a rounded version of it) appears in some tool result.

    Catches the agent stating a number it never actually computed."""
    for obs in observations:
        for n in numbers_in(obs):
            if abs(n - value) <= max(0.01, abs(value) * rel_tol):
                return True
    return False
