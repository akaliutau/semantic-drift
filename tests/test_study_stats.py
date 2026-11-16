import pytest

from selftrap.study_stats import holm,paired_binary,wilson


def test_exact_paired_test_and_case_denominators():
    a={str(i):True for i in range(10)}
    b={str(i):False for i in range(10)}
    r=paired_binary(a,b,samples=100)
    assert r['paired_n']==10 and r['a_only']==10 and r['b_only']==0
    assert r['exact_mcnemar_p']==2/1024
    assert r['bootstrap_95']==[1.,1.]
    assert wilson(0,40)[1]==pytest.approx(.0876216)
    assert holm({'a':.001,'b':.04,'c':.06})=={'a':.003,'b':.08,'c':.08}
