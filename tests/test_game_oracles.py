"""Independent small checks of the scientific ground truth, without API calls."""
from itertools import product
from selftrap.games.mastermind import feedback,inspect
from selftrap.programs import run_generator
from selftrap.games import blackbox,minesweeper,zendo
from selftrap.ordered_pilot import event_matches

def test_mastermind_duplicate_symbols_and_independent_feedback():
    assert feedback('AABB','AAAA')==[2,0]
    codes=[''.join(x) for x in product('ABC',repeat=3)]
    for secret in codes:
        for guess in codes:
            remaining=list(secret);misplaced=0
            for i,c in enumerate(guess):
                if c==secret[i]:remaining[i]=None
            for i,c in enumerate(guess):
                if c!=secret[i] and c in remaining:
                    misplaced+=1;remaining[remaining.index(c)]=None
            assert feedback(secret,guess)==[sum(a==b for a,b in zip(secret,guess)),misplaced]

def test_mastermind_identifiability_and_generator_execution():
    ambiguous=inspect({'alphabet':'ABC','secret':'AABB','guesses':['AAAA']})
    assert ambiguous['truth']=='UNKNOWN' and ambiguous['possible_count']>1
    solved=inspect({'alphabet':'ABC','secret':'AABB','guesses':['AABB']})
    assert solved['solutions']==['AABB'] and solved['truth']=='AABB'
    code="def generate(seed):\n p={'alphabet':'ABC','secret':'AABB','guesses':['AABB']}\n return {'truth':inspect(p)['truth']}"
    assert run_generator(code,5,'mastermind')['value']=={'truth':'AABB'}
    code="def generate(seed):\n random.seed(seed)\n return {'word':f'{chr(65)}{seed}'}"
    assert run_generator(code,5,'mastermind')['value']=={'word':'A5'}

def test_blackbox_physics_and_reciprocity():
    assert blackbox.ray(4,[],'T1')=='B1'
    assert blackbox.ray(4,[(0,1)],'T1')=='HIT'
    assert blackbox.ray(4,[(0,1)],'T0')=='REFLECT'
    assert blackbox.ray(4,[(2,2)],'T1')=='L1'
    for _,table in blackbox.world_table(4,3):
        for entry,out in table.items():
            if out not in ('HIT','REFLECT'):assert table[out]==entry

def test_minesweeper_safe_and_undetermined_are_distinct():
    safe=minesweeper.inspect({'size':4,'mines':[[0,0],[3,3]],'revealed':[[0,3]],'query':[0,2]})
    assert safe['truth']=='SAFE' and safe['possible_count']>1
    unknown=minesweeper.inspect({'size':4,'mines':[[0,0],[3,3]],'revealed':[[1,1]],'query':[0,1]})
    assert unknown['truth']=='UNKNOWN' and unknown['query_outcomes']==['MINE','SAFE']

def test_zendo_retains_counterexample_rule():
    r=zendo.inspect({'rule':[[1]],'examples':['000000','100000'],'query':'100001'})
    assert zendo.evaluate([[1]],'100001') and not zendo.evaluate([[1,-6]],'100001')
    assert r['truth']=='UNKNOWN' and r['query_outcomes']==[False,True]
    assert zendo.rule_mask([[1]]) in zendo.hypothesis_masks()
    assert zendo.rule_mask([[1,-6]]) in zendo.hypothesis_masks()

def test_epistemic_error_excludes_plain_inconsistent_answers():
    a=inspect({'alphabet':'ABC','secret':'ABAC','guesses':['AAAA','BBBB','ABCA','AABB']})
    assert a['solutions']==['ABAC','BACA']
    case={'analysis':a,'event_kind':'plausible_but_not_entailed'}
    assert event_matches(case,'BACA')
    assert not event_matches(case,'AABA') and not event_matches(case,'UNKNOWN')
    z=zendo.inspect({'rule':[[1]],'examples':['000000','100000'],'query':'100001'})
    assert event_matches({'analysis':z,'event_kind':'plausible_but_not_entailed'},'FALSE')
    assert not event_matches({'analysis':z,'event_kind':'plausible_but_not_entailed'},'UNKNOWN')
