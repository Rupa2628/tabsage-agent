"""Agent-loop tests with a scripted LLM: every recovery path is exercised deterministically."""
from tabsage.agent import Agent
from tabsage.llm import ScriptedLLM
from tabsage.memory import Memory
from tabsage.trace import Tracer

PLAN = {"plan": ["look at customers", "count rows", "answer"]}
COUNT = {"thought": "count", "action": "run_sql", "args": {"query": "SELECT COUNT(*) FROM customers"}}
DONE = {"thought": "done", "action": "final_answer", "args": {"answer": "There are 60 customers.", "value": 60}}


def make(db, responses, **kw):
    llm = ScriptedLLM(responses)
    return Agent(llm, db, tracer=Tracer(), **kw), llm


def test_happy_path(db):
    agent, llm = make(db, [PLAN, {"action": "describe_table", "args": {"table": "customers"}}, COUNT, DONE])
    res = agent.run("How many customers are there?")
    assert res.status == "success" and res.value == 60
    assert res.steps == 2 and res.tool_errors == 0
    assert res.plan == PLAN["plan"]
    # schema learned via describe_table is kept as a durable fact in later prompts
    assert "customers (60 rows)" in llm.calls[-1][1]["content"]


def test_recovers_from_invalid_json(db):
    agent, _ = make(db, [PLAN, "let me think about this...", COUNT, DONE])
    res = agent.run("How many customers?")
    assert res.status == "success" and res.parse_errors == 1


def test_recovers_from_sql_error(db):
    bad = {"action": "run_sql", "args": {"query": "SELECT COUNT(*) FROM customer"}}
    agent, llm = make(db, [PLAN, bad, COUNT, DONE])
    res = agent.run("How many customers?")
    assert res.status == "success" and res.tool_errors == 1
    assert "no such table" in llm.calls[2][1]["content"]  # error was shown to the model


def test_blocks_repeated_action(db):
    agent, _ = make(db, [PLAN, COUNT, COUNT, DONE])
    res = agent.run("How many customers?")
    assert res.status == "success" and res.loops_blocked == 1


def test_rejects_ungrounded_number_once(db):
    made_up = {"action": "final_answer", "args": {"answer": "There are 75 customers.", "value": 75}}
    agent, _ = make(db, [PLAN, made_up, COUNT, DONE])
    res = agent.run("How many customers?")
    assert res.status == "success" and res.value == 60 and res.ungrounded_rejections == 1


def test_replans_after_repeated_errors(db):
    bad = lambda t: {"action": "describe_table", "args": {"table": t}}
    agent, _ = make(db, [PLAN, bad("a"), bad("b"), bad("c"), {"plan": ["new plan"]}, COUNT, DONE])
    res = agent.run("How many customers?")
    assert res.status == "success" and res.replans == 1 and res.plan == ["new plan"]


def test_stops_at_step_limit(db):
    steps = [{"action": "run_sql", "args": {"query": f"SELECT {i}"}} for i in range(10)]
    agent, _ = make(db, [PLAN] + steps, max_steps=4)
    res = agent.run("Loop forever")
    assert res.status == "max_steps" and res.steps == 4


def test_checkpoint_and_resume(db, tmp_path):
    ckpt = tmp_path / "ckpt.json"
    # first run crashes (LLM runs out of responses) after one successful step
    agent, _ = make(db, [PLAN, COUNT], checkpoint_path=ckpt)
    res = agent.run("How many customers?")
    assert res.status == "error" and ckpt.exists()
    assert len(Memory.load(ckpt).steps) == 1

    # resume: no new plan call needed, continues from step 2
    agent2, llm2 = make(db, [DONE], checkpoint_path=ckpt)
    res2 = agent2.run("ignored", resume=True)
    assert res2.status == "success" and res2.value == 60
    assert res2.steps == 1  # the restored step; final_answer is not counted as a tool step
    assert len(llm2.calls) == 1
    assert not ckpt.exists()  # cleaned up after success


def test_memory_compacts_old_steps():
    m = Memory(question="q", keep_recent=2)
    for i in range(6):
        m.add_step(f"thought {i}", "run_sql", {"query": f"SELECT {i}"}, "x" * 400, ok=True)
    text = m.render()
    assert "thought 0" not in text and "thought 5" in text  # old thoughts dropped
    assert "[1] run_sql" in text                             # but old steps summarised
    assert len(text) < 2500
