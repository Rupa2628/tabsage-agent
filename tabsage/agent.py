"""The agent: plan, then loop think -> act -> observe until a grounded final answer.

    question
       |
    [planner] -> plan (3-6 steps)
       |
    loop (max_steps):
       render memory -> LLM -> parse JSON action
          |-- invalid JSON / unknown tool -> error fed back, retry
          |-- repeated identical action    -> blocked, told to try something else
          |-- tool call                    -> observation added to memory
          |-- final_answer                 -> grounding check -> done
       3 errors in a row -> replan with the errors as context
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .guardrails import LoopDetector, ParseError, is_grounded, parse_action, parse_plan
from .llm import LLM
from .memory import Memory
from .tools import TOOL_SPECS, Database, ToolError, call_tool
from .trace import Tracer

SYSTEM_PROMPT = """You are TabSage, a careful data analyst agent. You answer questions about tabular data \
by querying a read-only SQLite database.

TOOLS:
{tools}

RESPONSE FORMAT - reply with exactly ONE JSON object and nothing else:
{{"thought": "<short reasoning about what to do next>", "action": "<tool name>", "args": {{...}}}}

RULES:
1. Look at the schema (list_tables / describe_table) before writing SQL. Never guess column names.
2. Do ALL arithmetic in SQL. Never compute numbers in your head.
3. Only state numbers that appeared in a tool result.
4. If a tool returns an ERROR, read it and fix the cause; do not repeat the same call.
5. Finish with final_answer: "answer" is one clear sentence, "value" is the single key result \
(a number when the question asks for a quantity, otherwise a short string).
"""

PLANNER_PROMPT = """You plan analyses for a data agent. Available tables: {tables}.
Write a short plan (3-6 steps) to answer the question. Reply with JSON only: {{"plan": ["step 1", "step 2", ...]}}
{context}
QUESTION: {question}"""


@dataclass
class AgentResult:
    question: str
    status: str                      # "success" | "max_steps" | "error"
    answer: str | None = None
    value: object = None
    steps: int = 0
    tool_errors: int = 0
    parse_errors: int = 0
    loops_blocked: int = 0
    ungrounded_rejections: int = 0
    replans: int = 0
    llm_calls: int = 0
    seconds: float = 0.0
    plan: list[str] = field(default_factory=list)


class Agent:
    def __init__(self, llm: LLM, db: Database, max_steps: int = 12, max_consecutive_errors: int = 3,
                 tracer: Tracer | None = None, checkpoint_path: str | Path | None = None):
        self.llm = llm
        self.db = db
        self.max_steps = max_steps
        self.max_consecutive_errors = max_consecutive_errors
        self.tracer = tracer or Tracer()
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        tools = "\n".join(f"- {n}({json.dumps(s['args'])}): {s['description']}" for n, s in TOOL_SPECS.items())
        self.system_prompt = SYSTEM_PROMPT.format(tools=tools)

    # ------------------------------------------------------------------ planning
    def _plan(self, memory: Memory, result: AgentResult, context: str = "") -> None:
        prompt = PLANNER_PROMPT.format(tables=sorted(self.db.tables), question=memory.question, context=context)
        result.llm_calls += 1
        raw = self.llm.complete([{"role": "user", "content": prompt}])
        try:
            memory.plan = parse_plan(raw)
        except ParseError:
            # A failed plan is not fatal: the loop can still work step by step.
            memory.plan = ["Inspect the relevant tables", "Query with SQL", "Give the final answer"]
        self.tracer.log("plan", plan=memory.plan, context=context or None)

    # ------------------------------------------------------------------ main loop
    def run(self, question: str, resume: bool = False) -> AgentResult:
        t0 = time.monotonic()
        if resume and self.checkpoint_path and self.checkpoint_path.exists():
            memory = Memory.load(self.checkpoint_path)
            self.tracer.log("resume", from_step=len(memory.steps))
        else:
            memory = Memory(question=question)
        result = AgentResult(question=memory.question, status="max_steps")
        loops = LoopDetector()
        for s in memory.steps:  # rebuild loop history after a resume
            loops.is_repeat(_as_action(s))

        if not memory.plan:
            self._plan(memory, result)

        allowed = set(TOOL_SPECS)
        consecutive_errors = 0
        grounding_retry_used = False

        while len(memory.steps) < self.max_steps:
            messages = [{"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": memory.render()}]
            result.llm_calls += 1
            try:
                raw = self.llm.complete(messages)
            except Exception as e:
                self.tracer.log("llm_error", error=str(e))
                result.status = "error"
                break
            self.tracer.log("llm_reply", step=len(memory.steps) + 1, raw=raw)

            # 1. parse ---------------------------------------------------------
            try:
                action = parse_action(raw, allowed)
            except ParseError as e:
                result.parse_errors += 1
                consecutive_errors += 1
                memory.add_step("", "invalid_response", {}, f"{e} Reply with ONE JSON object in the required format.", ok=False)
                self._after_step(memory, result, consecutive_errors)
                if consecutive_errors >= self.max_consecutive_errors:
                    consecutive_errors = self._replan(memory, result)
                continue

            # 2. finish --------------------------------------------------------
            if action.name == "final_answer":
                answer = str(action.args.get("answer", "")).strip()
                value = action.args.get("value")
                if not answer:
                    consecutive_errors += 1
                    memory.add_step(action.thought, "final_answer", action.args,
                                    'final_answer needs a non-empty "answer".', ok=False)
                    self._after_step(memory, result, consecutive_errors)
                    continue
                num = _to_number(value)
                if num is not None and not grounding_retry_used and not is_grounded(num, memory.observations()):
                    # The number never appeared in any tool output: likely made up.
                    grounding_retry_used = True
                    result.ungrounded_rejections += 1
                    memory.add_step(action.thought, "final_answer", action.args,
                                    f"Rejected: the value {value} does not appear in any tool result. "
                                    "Compute it with run_sql, then answer again.", ok=False)
                    self.tracer.log("ungrounded_answer", value=value)
                    self._after_step(memory, result, consecutive_errors)
                    continue
                result.status, result.answer, result.value = "success", answer, value
                self.tracer.log("final_answer", answer=answer, value=value)
                break

            # 3. loop guard ----------------------------------------------------
            if loops.is_repeat(action):
                result.loops_blocked += 1
                consecutive_errors += 1
                memory.add_step(action.thought, action.name, action.args,
                                "Blocked: you already ran this exact action and have its result above. "
                                "Use that result or try a different approach.", ok=False)
                self._after_step(memory, result, consecutive_errors)
                if consecutive_errors >= self.max_consecutive_errors:
                    consecutive_errors = self._replan(memory, result)
                continue

            # 4. act -----------------------------------------------------------
            try:
                output = call_tool(self.db, action.name, action.args)
                if action.name == "describe_table":
                    memory.remember_schema(output["table"], output)
                memory.add_step(action.thought, action.name, action.args, json.dumps(output, default=str), ok=True)
                consecutive_errors = 0
            except ToolError as e:
                result.tool_errors += 1
                consecutive_errors += 1
                memory.add_step(action.thought, action.name, action.args, str(e), ok=False)
            self._after_step(memory, result, consecutive_errors)
            if consecutive_errors >= self.max_consecutive_errors:
                consecutive_errors = self._replan(memory, result)

        result.steps = len(memory.steps)
        result.replans = memory.replans
        result.plan = memory.plan
        result.seconds = round(time.monotonic() - t0, 2)
        self.tracer.log("done", **{k: v for k, v in result.__dict__.items() if k != "plan"})
        if self.checkpoint_path and result.status == "success":
            self.checkpoint_path.unlink(missing_ok=True)  # finished runs don't need resuming
        return result

    # ------------------------------------------------------------------ helpers
    def _after_step(self, memory: Memory, result: AgentResult, consecutive_errors: int) -> None:
        s = memory.steps[-1]
        self.tracer.log("step", index=s.index, action=s.action, args=s.args, ok=s.ok,
                        observation=s.observation[:500], consecutive_errors=consecutive_errors)
        if self.checkpoint_path:
            memory.save(self.checkpoint_path)

    def _replan(self, memory: Memory, result: AgentResult) -> int:
        memory.replans += 1
        errors = [s.observation for s in memory.steps if not s.ok][-3:]
        context = "The previous plan hit these errors, avoid them:\n" + "\n".join(f"- {e[:200]}" for e in errors)
        self._plan(memory, result, context)
        return 0


def _to_number(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", "").replace("%", "").strip())
        except ValueError:
            return None
    return None


def _as_action(step):
    from .guardrails import Action
    return Action(thought=step.thought, name=step.action, args=step.args)
