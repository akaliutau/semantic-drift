"""Independent small checks of the scientific ground truth, without API calls."""
from itertools import product
from selftrap.games.mastermind import feedback,inspect
from selftrap.programs import run_generator

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
