"""Bounded Zendo: semantic equivalence classes of small Boolean DNF rules."""
from functools import lru_cache
from itertools import combinations,product

RULES='''Zendo rule discovery on six Boolean attributes A,B,C,D,E,F.
Each scene is a six-character 0/1 string in that attribute order. The hidden
rule is an OR of one or two clauses. Each clause is an AND of one or two
literals on DISTINCT attributes; a literal can require its attribute to be 1
or to be 0. Clauses can overlap. Every rule in this grammar is possible;
there is no preference for simpler rules. Labeled scenes are truthful.
For the query scene answer TRUE if every rule consistent with ALL labeled
scenes labels it true, FALSE if every such rule labels it false, otherwise
UNKNOWN. Matching one plausible rule is insufficient. Submit through
submit_answer(answer=...). No external execution tools.'''

AUTHOR='''generate(seed) builds puzzle={"rule":[[1,-2],[3,4]],
"examples":["000000","110000",...],"query":"111000"}.
rule is DNF: [[1,-2],[3,4]] means (A and not B) or (C and D). Exactly six
attributes. Each rule has 1 or 2 distinct clauses, each 1 or 2 signed literals
from +/-1..6, with distinct attributes inside each clause. 2..18 distinct
example scenes; query is another scene. inspect(puzzle) computes labels and
enumerates ALL semantically distinct rules in this grammar. It returns
public, truth (TRUE/FALSE/UNKNOWN), possible_count and query_outcomes.
Your program chooses the rule, examples, query and exact trap_answer.
Target a specific quantifier/negation/conjunction or premature-rule mistake.
Use text scenes only; perception is outside this test.'''

SCENES=[''.join(x) for x in product('01',repeat=6)]

def evaluate(rule,scene):
    return any(all(scene[abs(literal)-1]==('1' if literal>0 else '0') for literal in clause) for clause in rule)

def rule_mask(rule):
    return sum((1<<i) for i,scene in enumerate(SCENES) if evaluate(rule,scene))

@lru_cache(maxsize=1)
def hypothesis_masks():
    literals=[i for j in range(1,7) for i in (j,-j)]
    clauses=[(x,) for x in literals]+[x for x in combinations(literals,2) if abs(x[0])!=abs(x[1])]
    masks=[rule_mask([clause]) for clause in clauses]
    return sorted(set(masks)|{a|b for a,b in combinations(masks,2)})

def inspect(puzzle):
    rule=puzzle['rule'];examples=puzzle['examples'];query=puzzle['query']
    if not isinstance(rule,list) or len(rule) not in (1,2):raise ValueError('Rule needs one or two clauses')
    for clause in rule:
        if not isinstance(clause,list) or len(clause) not in (1,2) or any(type(x) is not int or not 1<=abs(x)<=6 for x in clause) or len({abs(x) for x in clause})!=len(clause):raise ValueError('Invalid clause')
    if len({tuple(sorted(c)) for c in rule})!=len(rule):raise ValueError('Repeated clause')
    if not isinstance(examples,list) or not 2<=len(examples)<=18 or len(set(examples))!=len(examples) or any(x not in SCENES for x in examples) or query not in SCENES or query in examples:raise ValueError('Invalid examples/query')
    labels=[evaluate(rule,x) for x in examples];indices=[SCENES.index(x) for x in examples];q=SCENES.index(query)
    matches=[mask for mask in hypothesis_masks() if all(bool(mask&(1<<i))==label for i,label in zip(indices,labels))]
    outcomes={bool(mask&(1<<q)) for mask in matches}
    return {'public':{'attributes':['A','B','C','D','E','F'],'examples':[{'scene':s,'label':label} for s,label in zip(examples,labels)],'query':query},
        'truth':('TRUE' if next(iter(outcomes)) else 'FALSE') if len(outcomes)==1 else 'UNKNOWN',
        'possible_count':len(matches),'query_outcomes':sorted(outcomes),'allowed_answers':['TRUE','FALSE','UNKNOWN']}
