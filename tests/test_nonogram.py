from selftrap.nonogram import puzzle,solve
from selftrap.programs import run_generator,validate_source
import pytest

def test_real_nonogram_solver_and_pure_generator():
    bitmap=['00100','01110','11111','01110','00100']
    result=solve(puzzle(bitmap))
    assert result['status']=='COMPLETE' and result['solutions']==[bitmap]
    code="def generate(seed):\n    return ['00100','01110','11111','01110','00100']\n"
    assert run_generator(code,7)['value']==bitmap
    with pytest.raises(ValueError):validate_source('import os\ndef generate(seed):\n return os.environ\n')
