from types import SimpleNamespace
import pytest
from selftrap.parsing import submission

def test_explicit_terminal_call_preserves_the_action_and_rejects_drafts():
    tool=[{'name':'submit_cell'}]
    r=SimpleNamespace(calls=[],text='I infer zero.\nsubmit_cell(value=0)')
    assert submission(r,tool)==({'value':0},'terminal_call')
    r.text+='\nMaybe something else.'
    with pytest.raises((ValueError,SyntaxError)):submission(r,tool)
    r.text='another_tool(value=0)'
    with pytest.raises(ValueError):submission(r,tool)
