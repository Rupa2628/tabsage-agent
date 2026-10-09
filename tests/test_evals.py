from pathlib import Path

from tabsage.agent import AgentResult
from tabsage.evals import gold_answer, load_tasks, score

TASKS = Path(__file__).resolve().parent.parent / "evals" / "tasks.json"


def test_every_gold_query_returns_one_value(db):
    tasks = load_tasks(TASKS)
    assert len(tasks) >= 15
    assert len({t["id"] for t in tasks}) == len(tasks)
    for t in tasks:
        assert gold_answer(db, t["gold_sql"]) is not None, t["id"]


def r(value=None, answer="", status="success"):
    return AgentResult(question="q", status=status, value=value, answer=answer)


def test_score_numbers():
    assert score(r(value=60), 60)
    assert score(r(value="1,234,567"), 1234567)
    assert score(r(value=None, answer="Revenue was 1234567.0"), 1234567)
    assert not score(r(value=61), 60)
    assert not score(r(value=60, status="max_steps"), 60)


def test_score_strings():
    assert score(r(value="Hyderabad"), "Hyderabad")
    assert score(r(value=None, answer="The top city is hyderabad."), "Hyderabad")
    assert not score(r(value="Pune"), "Hyderabad")
