from pathlib import Path

import pytest

from tabsage.tools import Database

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def db():
    return Database(DATA)
