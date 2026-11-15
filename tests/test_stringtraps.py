from selftrap.stringtraps import compile_string

def test_program_counts_are_mechanical_and_do_not_depend_on_author_claims():
    p={'parts':[{'text':'abba','repeat':5},{'text':'r','repeat':2}],'character':'b',
       'predicted_wrong':9,'p_wrong':[.5,.5],'diagnosis':'test'}
    assert compile_string(p)['true']==10
