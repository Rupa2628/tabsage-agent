# TabSage

**A plan-and-act agent that answers natural-language questions over CSV data with SQL, with built-in guardrails and an evaluation harness.**

Author: **Rupa Srivalli**

Example run (output format; your model's plan wording will differ):

```
$ python -m tabsage ask "Which product category generated the most revenue from completed orders?"

Plan:
  1. Inspect the orders and products tables
  2. Join orders to products on product_id
  3. Filter to completed orders and sum quantity x price per category
  4. Return the top category

Status: success  |  steps: 4  |  tool errors: 0
Answer: Electronics generated the most revenue from completed orders.
```

## Why

LLMs are bad at arithmetic and like to invent numbers. TabSage never lets the model do math in its head. It makes the model write SQL, runs the SQL in a locked read-only database, and **rejects any final number that never appeared in a tool result**. Every step is logged as a JSON trace so a run can be audited afterwards.

## Architecture

```
            question
               |
          +---------+
          | Planner |  -> 3-6 step plan (JSON)
          +---------+
               |
   +-----------v------------------------------------------+
   |  Reasoning loop (max 12 steps)                        |
   |                                                       |
   |  memory.render() -> LLM -> JSON action                |
   |        |                                              |
   |        +-- parse error ---------> error fed back      |
   |        +-- repeated action -----> blocked             |
   |        +-- tool call -----------> observation         |
   |        +-- final_answer --------> grounding check     |
   |                                                       |
   |  3 errors in a row -> replan with errors as context   |
   +-------------------------------------------------------+
        |               |                 |
   +---------+    +-----------+     +-----------+
   |  Tools  |    |  Memory   |     |  Tracer   |
   | SQLite  |    | plan      |     | JSONL     |
   | (r/o)   |    | facts     |     | per run   |
   +---------+    | steps     |     +-----------+
                  | checkpoint|
                  +-----------+
```

| Component | File | What it does |
|---|---|---|
| Agent loop + planner | `tabsage/agent.py` | Plans, then think → act → observe until a grounded answer or the step limit |
| Tools | `tabsage/tools.py` | `list_tables`, `describe_table`, `run_sql` over an in-memory SQLite DB |
| Guardrails | `tabsage/guardrails.py` | JSON action parsing, SQL safety, loop detection, answer grounding |
| Memory | `tabsage/memory.py` | Plan, durable facts, compacted step history, atomic JSON checkpoints |
| Tracing | `tabsage/trace.py` | One JSON line per event for replay and auditing |
| Eval harness | `tabsage/evals.py`, `evals/tasks.json` | 15 tasks, gold answers computed from gold SQL, metrics report |
| LLM client | `tabsage/llm.py` | Any OpenAI-compatible API (Groq, Gemini, OpenRouter, Ollama) + a scripted LLM for tests |

No agent framework is used. The loop is written by hand so every decision is visible and testable.

## Guardrails and failure handling

| Failure mode | Mitigation |
|---|---|
| Model invents a number | Grounding check: a numeric answer must appear in some tool result, otherwise it is rejected and the model is told to compute it with SQL |
| Model does math in its head | System prompt rule plus grounding check |
| Malformed output (prose, broken JSON, unknown tool) | Tolerant parser (handles code fences and chatter); the error is fed back and the model retries |
| Hallucinated column / table names | Must inspect schema first; `describe_table` errors list the real tables; SQL errors are fed back verbatim |
| Destructive SQL | Two layers: validator allows one `SELECT`/`WITH` statement only, and the DB runs with `PRAGMA query_only = ON` |
| Runaway queries (accidental cross joins) | Per-query time limit via SQLite progress handler, 50-row result cap |
| Agent repeats the same call | Loop detector blocks identical actions |
| Agent stuck on a bad approach | 3 consecutive errors trigger a replan that includes the errors |
| Never finishes | Hard step limit (default 12) |
| Context blow-up on long runs | Observations truncated, old steps compressed to one line, schemas kept as durable facts |
| Crash mid-run (rate limit, network) | LLM retries with exponential backoff; memory checkpointed after every step; `--resume` continues |

## Quick start

```bash
git clone https://github.com/<your-username>/tabsage-agent.git
cd tabsage-agent
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # then put your API key in .env
python -m tabsage ask "How many customers have never placed an order?"
```

Any OpenAI-compatible provider works. Free options: [Groq](https://console.groq.com/keys) (default in `.env.example`) or [Google AI Studio](https://aistudio.google.com/apikey) (see the commented lines in `.env.example`).

Use your own data: `python -m tabsage --data path/to/csv_folder ask "..."`

## Evaluation

```bash
python -m tabsage eval
```

Runs the 15 tasks in `evals/tasks.json` (lookup, filter, aggregation, join, multi-step). Each task stores a **gold SQL query** instead of a hand-typed answer; the expected value is computed by running it on the same data, so the expected answers can't drift from the data.

A task passes only if the run finished with `final_answer` and the value matches (numbers within 0.5%; strings case-insensitive). The report (`evals/results/<label>.md` and `.json`) records:

- accuracy overall and per category, completion rate
- average steps and latency
- tool errors per step, parse errors, blocked loops
- ungrounded-answer rejections, replans, runs that hit the step limit

Per-task traces go to `evals/results/traces/` so every failure can be inspected step by step.

### Results

_Run `python -m tabsage eval` with your model and paste the summary table here._

## Tests

```bash
pytest -q
```

36 tests, no API key needed. A scripted LLM replays fixed responses so every recovery path (bad JSON, SQL error, repeated action, made-up number, replan, step limit, crash + resume) is tested deterministically. The tests also check that every gold SQL query returns exactly one value.

## Project layout

```
tabsage/            agent package
  agent.py          planner + reasoning loop
  tools.py          SQLite tools
  guardrails.py     parsing, SQL safety, loop detection, grounding
  memory.py         working memory + checkpoints
  trace.py          JSONL tracing
  evals.py          eval harness
  llm.py            LLM clients
evals/tasks.json    eval tasks with gold SQL
data/               synthetic sample dataset (scripts/make_data.py)
tests/              pytest suite
```

## Limitations and next steps

- Single-agent, text-JSON actions. Native function calling would reduce parse errors on models that support it.
- The grounding check covers numeric answers only; string answers are not yet verified against tool output.
- 15 eval tasks on one synthetic dataset is a small benchmark. Next: more tasks, messier real-world CSVs, and comparing models.
- Add a human-in-the-loop confirmation step for expensive or ambiguous queries.

## License

MIT © Rupa Srivalli
