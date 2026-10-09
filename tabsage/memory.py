"""Working memory and checkpointing.

Three layers:
  * plan     - the step list from the planner, rewritten when the agent replans
  * facts    - durable notes (table schemas the agent has looked up). Never compacted.
  * steps    - the action/observation history. Recent steps are shown in full, older
               ones are compressed to one line each so the prompt stays bounded.

The whole state is saved to a JSON checkpoint after every step, so a long run that
crashes (rate limit, network) can resume from where it stopped.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Step:
    index: int
    thought: str
    action: str
    args: dict
    observation: str
    ok: bool


@dataclass
class Memory:
    question: str
    plan: list[str] = field(default_factory=list)
    facts: dict[str, str] = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)
    replans: int = 0
    keep_recent: int = 4
    max_obs_chars: int = 1500

    # -------------------------------------------------------------- updates
    def add_step(self, thought: str, action: str, args: dict, observation: str, ok: bool) -> Step:
        if len(observation) > self.max_obs_chars:
            observation = observation[: self.max_obs_chars] + f"... [truncated, {len(observation)} chars total]"
        step = Step(len(self.steps) + 1, thought, action, args, observation, ok)
        self.steps.append(step)
        return step

    def remember_schema(self, table: str, described: dict) -> None:
        cols = ", ".join(f"{c['name']}:{c['dtype']}" + (f" {c['values']}" if "values" in c else "")
                         for c in described["columns"])
        self.facts[f"schema:{table}"] = f"{table} ({described['rows']} rows): {cols}"

    # -------------------------------------------------------------- rendering
    def render(self) -> str:
        parts = [f"QUESTION: {self.question}"]
        if self.plan:
            parts.append("PLAN:\n" + "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(self.plan)))
        if self.facts:
            parts.append("KNOWN FACTS:\n" + "\n".join(f"  - {v}" for v in self.facts.values()))
        if self.steps:
            old, recent = self.steps[:-self.keep_recent], self.steps[-self.keep_recent:]
            lines = []
            for s in old:  # compressed
                status = "ok" if s.ok else "ERROR"
                lines.append(f"  [{s.index}] {s.action}({_short(s.args)}) -> {status}: {s.observation[:100]}")
            for s in recent:  # full
                lines.append(f"  [{s.index}] thought: {s.thought}\n"
                             f"      action: {s.action} {json.dumps(s.args)}\n"
                             f"      result ({'ok' if s.ok else 'ERROR'}): {s.observation}")
            parts.append("HISTORY:\n" + "\n".join(lines))
        parts.append(f"You have used {len(self.steps)} steps. Reply with the next action as JSON.")
        return "\n\n".join(parts)

    def observations(self, ok_only: bool = True) -> list[str]:
        return [s.observation for s in self.steps if s.ok or not ok_only]

    # -------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        tmp.replace(path)  # atomic: a crash mid-write never leaves a half checkpoint

    @classmethod
    def load(cls, path: str | Path) -> "Memory":
        data = json.loads(Path(path).read_text())
        data["steps"] = [Step(**s) for s in data["steps"]]
        return cls(**data)


def _short(args: dict) -> str:
    s = json.dumps(args)
    return s if len(s) <= 80 else s[:77] + "..."
