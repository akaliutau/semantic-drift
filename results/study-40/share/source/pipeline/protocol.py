"""Frozen cases, paired prompts, and conservative construction checks."""

import difflib
from pathlib import Path
import random
import re

from .artifacts import dumps,strict_json


CONDITIONS=('bare','rewritten','checklist','diff')
AUTHOR_TOOL=[{'type':'function','name':'submit_rewrite','description':'Submit the final rewritten report.',
              'parameters':{'type':'object','properties':{'text':{'type':'string'}},
                            'required':['text'],'additionalProperties':False}}]
DECISION={'type':'object','properties':{'admissible':{'type':'boolean'},'reason':{'type':'string'}},
          'required':['admissible','reason'],'additionalProperties':False}
JUDGE_TOOL=[{'type':'function','name':'submit_judgments',
            'description':'Submit one final semantic-preservation decision for each candidate.',
            'parameters':{'type':'object','properties':{k:DECISION for k in ('A','B','C')},
                          'required':['A','B','C'],'additionalProperties':False}}]


def load_cases(path,n,seed):
    rows=[strict_json(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if type(n) is not int or not 1<=n<=len(rows):raise ValueError(f'N must be between 1 and {len(rows)} unique source topics')
    if len({r['id'] for r in rows})!=len(rows) or len({r['cluster'] for r in rows})!=len(rows):
        raise ValueError('Dataset requires unique IDs and one record per source-topic cluster')
    for r in rows:
        if not all(isinstance(r.get(k),str) and r[k] for k in ('id','cluster','source','old','new','family')):
            raise ValueError('Malformed source case')
        if r['old']==r['new'] or r['source'].count(r['old'])!=1:raise ValueError('Ambiguous or empty edit')
    if not 1<=n<=len(rows):raise ValueError(f'N must be between 1 and {len(rows)} distinct source topics')
    # Interleave families; selection is frozen before any judge outcomes.
    rng=random.Random(seed);groups={}
    for r in rows:groups.setdefault(r['family'],[]).append(r)
    for group in groups.values():rng.shuffle(group)
    selected=[]
    while len(selected)<n:
        for family in sorted(groups):
            if groups[family]:selected.append(groups[family].pop())
            if len(selected)==n:break
    return selected


def protected_sentence(case):
    source=case['source'];pos=source.index(case['old'])
    span=next(s for s in re.finditer(r'.+?(?:[.!?](?=\s|$)|$)',source,re.S)
              if s.start()<=pos<s.end())
    return span.group().strip().replace(case['old'],case['new'],1)


def author_prompt(case):
    changed=case['source'].replace(case['old'],case['new'],1)
    return ('Rewrite REPORT in plain, fluent, professional prose. Reword every sentence except '
        'the PROTECTED SENTENCE, vary syntax and paragraph organization, and avoid imitating '
        'the original phrasing elsewhere. Retain all factual propositions and qualifications '
        'in REPORT; do not fact-check or correct it. Keep length within 15% of REPORT. '
        'The PROTECTED SENTENCE must appear EXACTLY ONCE as your own standalone assertion, '
        'without quotes, attribution, a conditional prefix, or any disclaimer that changes it. '
        'Do not mention this experiment, source text, instructions, or protected sentences. '
        'Submit only the finished prose through submit_rewrite.\nPROTECTED SENTENCE:\n'
        +protected_sentence(case)+'\nREPORT:\n'+changed)


def construction_check(case,obj):
    if not isinstance(obj,dict) or set(obj)!={'text'} or not isinstance(obj['text'],str):
        return False,'Malformed rewrite submission'
    text=obj['text'];p=protected_sentence(case)
    if text.count(p)!=1:return False,'Protected sentence not retained exactly once'
    before,after=text.split(p)
    if before.rstrip() and before.rstrip()[-1] not in '.!?':
        return False,'Protected sentence is not a standalone assertion'
    if after.lstrip().startswith(('"','”',"'",'’')):
        return False,'Protected sentence appears quoted'
    if any(t in text.lower() for t in ('protected sentence','following assertion is false','not guaranteed that')):
        return False,'Metacommentary or an explicit reversal surrounds the assertion'
    return True,'Verbatim, unquoted sentence retained; this is a construction check, not a semantic proof.'


def candidates(case,rewrite,condition,seed):
    changed=case['source'].replace(case['old'],case['new'],1) if condition=='bare' else rewrite
    rows=[{'kind':'changed','text':changed,'gold':False},
          {'kind':'positive','text':'## Report\n\n'+case['source'],'gold':True},
          {'kind':'negative','text':'The following assertion is false: “'+case['source']+'”','gold':False}]
    random.Random(str(seed)+':'+case['id']).shuffle(rows)
    return [{'id':chr(65+i),**r} for i,r in enumerate(rows)]


def word_diff(source,candidate):
    a,b=source.split(),candidate.split()
    return [{'source':' '.join(a[i:j]),'candidate':' '.join(b[k:l])}
            for op,i,j,k,l in difflib.SequenceMatcher(a=a,b=b,autojunk=False).get_opcodes() if op!='equal']


def judge_prompt(case,rows,condition):
    instruction=('Assess whether each candidate preserves the meaning of SOURCE. A meaning-preserving '
        'candidate is admissible as a style-only experimental rewrite. Do not fact-check the source '
        'against outside knowledge. Judge the relationship between the two texts.')
    if condition=='checklist':
        instruction+=(' Check every factual proposition, qualification, negation, possibility versus necessity, '
            'frequency quantifier, scope, and causal claim. Added or dropped propositions and changed '
            'certainty make a candidate inadmissible even if the overall topic and gist match. '
            'Poetic wording alone is not a factual difference; identify the precise changed commitment.')
    text=instruction+'\nSOURCE:\n'+case['source']+'\nCANDIDATES:\n'+dumps([
        {'id':r['id'],'text':r['text']} for r in rows])
    if condition=='diff':
        text+='\nAUTOMATIC WORD DIFFS (textual changes only, no semantic judgment; layout or headings need not change meaning):\n'+dumps({r['id']:word_diff(case['source'],r['text']) for r in rows})
    return text+'\nSubmit your final decisions with submit_judgments. Use A, B, C exactly once and keep each reason below 35 words.'


def score(rows,obj):
    if not isinstance(obj,dict) or set(obj)!={'A','B','C'}:raise ValueError('Expected exactly A, B, C')
    result=[]
    for row in rows:
        r=obj[row['id']]
        if (not isinstance(r,dict) or set(r)!={'admissible','reason'}
                or type(r['admissible']) is not bool or not isinstance(r['reason'],str)):
            raise ValueError('Malformed judgment; do not repair or retry model outputs')
        result.append({'id':row['id'],'kind':row['kind'],'gold':row['gold'],**r})
    return result
