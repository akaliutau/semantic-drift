import pytest
from selftrap.game import Level


@pytest.fixture
def level():
    return Level.from_rows(["#######", "#_____#", "#_.$__#", "#_____#",
                            "#_@$._#", "#_____#", "#######"])
