import pytest

from tabsage.guardrails import (Action, LoopDetector, ParseError, is_grounded, parse_action, parse_plan)

ALLOWED = {"list_tables", "run_sql", "final_answer"}


def test_parse_plain_json():
    a = parse_action('{"thought": "t", "action": "run_sql", "args": {"query": "SELECT 1"}}', ALLOWED)
    assert a.name == "run_sql" and a.args["query"] == "SELECT 1"


def test_parse_fenced_json_with_chatter():
    text = 'Sure! Here you go:\n```json\n{"action": "list_tables", "args": {}}\n```'
    assert parse_action(text, ALLOWED).name == "list_tables"


@pytest.mark.parametrize("text", [
    "I think I should list the tables.",
    '{"thought": "no action here"}',
    '{"action": "delete_everything", "args": {}}',
    '{"action": "run_sql", "args": "SELECT 1"}',
])
def test_parse_rejects_bad_replies(text):
    with pytest.raises(ParseError):
        parse_action(text, ALLOWED)


def test_parse_plan():
    assert parse_plan('{"plan": ["a", "b"]}') == ["a", "b"]
    with pytest.raises(ParseError):
        parse_plan('{"plan": []}')


def test_loop_detector_ignores_whitespace_and_case():
    d = LoopDetector()
    assert not d.is_repeat(Action("", "run_sql", {"query": "SELECT 1"}))
    assert d.is_repeat(Action("", "run_sql", {"query": "select   1"}))
    assert not d.is_repeat(Action("", "run_sql", {"query": "SELECT 2"}))


def test_grounding():
    obs = ['{"rows": [[1234567.0]]}', '{"rows": [[12.35]]}']
    assert is_grounded(1234567, obs)
    assert is_grounded(12.35, obs)
    assert not is_grounded(999, obs)
