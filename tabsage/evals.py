"""Evaluation harness.

Each task has a question and a *gold SQL* query. The expected answer is computed by
running the gold SQL on the same data, so expected answers are never hand-typed and
stay correct if the dataset changes.

For every task we record: correctness, final status, steps, tool errors, parse
errors, blocked loops, ungrounded-answer rejections, replans and latency. Results
are written to evals/results/ as JSON (machine) and Markdown (human).
"""
from __future__ import annotations

import json
import re
import statistics
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from .agent import Agent, AgentResult
from .guardrails import numbers_in
from .tools import Database
from .trace import Tracer


def load_tasks(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())


def gold_answer(db: Database, gold_sql: str):
    rows = db.run_sql(gold_sql)["rows"]
    if len(rows) != 1 or len(rows[0]) != 1:
        raise ValueError(f"Gold SQL must return exactly one value: {gold_sql}")
    return rows[0][0]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def score(result: AgentResult, expected) -> bool:
    """Numbers: within 0.5% (or 0.01). Strings: exact match on value, or contained in answer."""
    if result.status != "success":
        return False
    if isinstance(expected, (int, float)):
        candidates = []
        if isinstance(result.value, (int, float)) and not isinstance(result.value, bool):
            candidates.append(float(result.value))
        elif result.value is not None:
            candidates += numbers_in(str(result.value))
        if not candidates:  # fall back to numbers in the sentence
            candidates = numbers_in(result.answer or "")
        tol = max(0.01, abs(expected) * 0.005)
        return any(abs(c - expected) <= tol for c in candidates)
    exp = _norm(expected)
    return _norm(result.value or "") == exp or exp in _norm(result.answer or "")


def run_eval(make_agent: Callable[[Tracer], Agent], db: Database, tasks: list[dict],
             out_dir: str | Path = "evals/results", label: str = "run") -> dict:
    out = Path(out_dir)
    (out / "traces").mkdir(parents=True, exist_ok=True)
    rows = []
    for task in tasks:
        expected = gold_answer(db, task["gold_sql"])
        tracer = Tracer(out / "traces" / f"{label}_{task['id']}.jsonl")
        agent = make_agent(tracer)
        try:
            res = agent.run(task["question"])
        except Exception as e:  # a crash is a failed task, not a crashed eval
            res = AgentResult(question=task["question"], status="error", answer=str(e))
        ok = score(res, expected)
        rows.append({"id": task["id"], "category": task["category"], "question": task["question"],
                     "expected": expected, "correct": ok, **asdict(res)})
        print(f"[{'PASS' if ok else 'FAIL'}] {task['id']} {task['question'][:60]:<60} "
              f"steps={res.steps} status={res.status}")

    summary = summarise(rows)
    report = {"label": label, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "summary": summary, "tasks": rows}
    (out / f"{label}.json").write_text(json.dumps(report, indent=2, default=str))
    (out / f"{label}.md").write_text(to_markdown(report))
    return report


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["correct"])
    total_steps = sum(r["steps"] for r in rows) or 1
    return {
        "tasks": n,
        "accuracy": round(sum(r["correct"] for r in rows) / n, 3),
        "accuracy_by_category": {c: round(sum(v) / len(v), 3) for c, v in sorted(by_cat.items())},
        "completion_rate": round(sum(r["status"] == "success" for r in rows) / n, 3),
        "avg_steps": round(statistics.mean(r["steps"] for r in rows), 2),
        "avg_seconds": round(statistics.mean(r["seconds"] for r in rows), 2),
        "tool_error_rate_per_step": round(sum(r["tool_errors"] for r in rows) / total_steps, 3),
        "parse_errors": sum(r["parse_errors"] for r in rows),
        "loops_blocked": sum(r["loops_blocked"] for r in rows),
        "ungrounded_rejections": sum(r["ungrounded_rejections"] for r in rows),
        "replans": sum(r["replans"] for r in rows),
        "hit_step_limit": sum(r["status"] == "max_steps" for r in rows),
    }


def to_markdown(report: dict) -> str:
    s = report["summary"]
    lines = [f"# Eval report: {report['label']}", f"_{report['timestamp']}_", "",
             "| Metric | Value |", "|---|---|"]
    for k, v in s.items():
        if k != "accuracy_by_category":
            lines.append(f"| {k} | {v} |")
    lines += ["", "**Accuracy by category**", "", "| Category | Accuracy |", "|---|---|"]
    lines += [f"| {c} | {a} |" for c, a in s["accuracy_by_category"].items()]
    lines += ["", "**Per task**", "", "| id | correct | status | steps | expected | agent value |", "|---|---|---|---|---|---|"]
    for t in report["tasks"]:
        lines.append(f"| {t['id']} | {'yes' if t['correct'] else 'no'} | {t['status']} | {t['steps']} "
                     f"| {t['expected']} | {t['value']} |")
    return "\n".join(lines) + "\n"
