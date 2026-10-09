"""Command line interface.

    python -m tabsage ask "Which city has the most customers?"
    python -m tabsage ask "..." --resume          # continue a crashed run
    python -m tabsage eval                         # run the evaluation harness
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from .agent import Agent
from .evals import load_tasks, run_eval
from .llm import OpenAICompatibleLLM
from .tools import Database
from .trace import Tracer


def load_dotenv(path: str = ".env") -> None:
    """Tiny .env loader so we don't need an extra dependency."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(prog="tabsage", description="Plan-and-act data analyst agent over CSV files.")
    p.add_argument("--data", default="data", help="folder of CSV files (default: data)")
    p.add_argument("--max-steps", type=int, default=12)
    sub = p.add_subparsers(dest="cmd", required=True)

    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question")
    ask.add_argument("--resume", action="store_true", help="resume from the last checkpoint")

    ev = sub.add_parser("eval", help="run the evaluation harness")
    ev.add_argument("--tasks", default="evals/tasks.json")
    ev.add_argument("--label", default=None, help="name for this run's report")
    args = p.parse_args(argv)

    db = Database(args.data)
    llm = OpenAICompatibleLLM()

    if args.cmd == "ask":
        tracer = Tracer(Path("traces") / f"ask_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
        agent = Agent(llm, db, max_steps=args.max_steps, tracer=tracer, checkpoint_path=".tabsage/checkpoint.json")
        res = agent.run(args.question, resume=args.resume)
        print("\nPlan:")
        for i, s in enumerate(res.plan, 1):
            print(f"  {i}. {s}")
        print(f"\nStatus: {res.status}  |  steps: {res.steps}  |  tool errors: {res.tool_errors}  |  {res.seconds}s")
        print(f"Answer: {res.answer}")
        print(f"Trace:  {tracer.path}")
        return 0 if res.status == "success" else 1

    label = args.label or f"{llm.model.replace('/', '_')}_{time.strftime('%Y%m%d_%H%M%S')}"
    report = run_eval(lambda tr: Agent(llm, db, max_steps=args.max_steps, tracer=tr),
                      db, load_tasks(args.tasks), label=label)
    s = report["summary"]
    print(f"\nAccuracy {s['accuracy']:.0%} on {s['tasks']} tasks | avg steps {s['avg_steps']} | "
          f"report: evals/results/{label}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
