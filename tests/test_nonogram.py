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

def test_hint_information_equivalence_is_certified_by_a_real_automorphism():
    from selftrap.nonogram import validate_symmetric_case,actor_view
    obj={'bitmap':['00100','01110','11111','01110','00100'],'query':[2,2],
         'predicted_wrong':0,'diagnostic_hint':[1,0],'control_hint':[1,4],
         'p_wrong':[.2,.7],'diagnosis':'A predicted left-right evidence-use asymmetry.',
         'hint_forecasts':{m:{'diagnostic_correct':.8,'control_correct':.3} for m in ['qwen3.5-397b','gemma-4-31b-it']}}
    case=validate_symmetric_case(obj)
    a=actor_view(case,'diagnostic');b=actor_view(case,'control')
    assert case['symmetry_certificate']['query_fixed']
    assert a['puzzle']==b['puzzle'] and a['query']==b['query']
    assert a['revealed_cells'][0][2]==b['revealed_cells'][0][2]
