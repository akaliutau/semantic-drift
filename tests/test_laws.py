import pytest
from selftrap.laws import compile_law

def test_law_predictions_cannot_be_ground_truth_or_out_of_scope():
    law={'prefix':'aa','atom':'banaba','suffix':'a','character':'a',
         'scale':2,'offset':1,'p_exact':[.5,.5],'mechanism':'test'}
    row=compile_law(law,3)
    assert row['true']==12 and row['predicted']==7
    law.update(scale=3,offset=3)
    with pytest.raises(ValueError,match='wrong count'):compile_law(law,3)
