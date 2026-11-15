"""Mastermind candidate enumeration; repeated colors use multiset feedback."""
from collections import Counter
from itertools import product

RULES='''Mastermind. A hidden code uses the stated alphabet, with repetitions
allowed. Each clue gives a guess, the count of symbols in the correct position
(exact), and the count correct but misplaced (misplaced). Each occurrence can
be matched only once. No preference for simpler or more likely codes applies.
Submit the unique code if the evidence determines it; otherwise submit UNKNOWN.
Use submit_answer(answer=...). You have no external execution tools.'''

AUTHOR='''generate(seed) builds puzzle={"alphabet":"ABCDE", "secret":"AABCD",
"guesses":["AACDE", ...]}. Length 4 or 5; alphabet 3..5 distinct uppercase
letters; at most 12 guesses; alphabet_size**length <= 4096. Repetitions allowed.
inspect(puzzle) computes truthful feedback, all consistent solutions, the correct
answer (unique code or UNKNOWN), and public clues. Your program chooses the
secret, guesses, and trap_answer. Do not hand-compute feedback. Seek a specific
mistake combining clues or accounting for repetitions; size alone is weak.'''

def feedback(secret,guess):
    exact=sum(a==b for a,b in zip(secret,guess))
    return [exact,sum((Counter(secret)&Counter(guess)).values())-exact]

def inspect(puzzle):
    alphabet=puzzle['alphabet'];secret=puzzle['secret'];guesses=puzzle['guesses']
    if not isinstance(alphabet,str) or not 3<=len(alphabet)<=5 or len(set(alphabet))!=len(alphabet) or not alphabet.isascii() or not alphabet.isupper():raise ValueError('Alphabet must be 3..5 distinct uppercase ASCII letters')
    n=len(secret)
    if n not in (4,5) or len(alphabet)**n>4096 or any(c not in alphabet for c in secret):raise ValueError('Invalid code length/symbols')
    if not isinstance(guesses,list) or not 1<=len(guesses)<=12 or any(not isinstance(g,str) or len(g)!=n or any(c not in alphabet for c in g) for g in guesses):raise ValueError('Invalid guesses')
    clues=[{'guess':g,'exact':feedback(secret,g)[0],'misplaced':feedback(secret,g)[1]} for g in guesses]
    candidates=[''.join(c) for c in product(alphabet,repeat=n)]
    solutions=[s for s in candidates if all(feedback(s,c['guess'])==[c['exact'],c['misplaced']] for c in clues)]
    return {'public':{'alphabet':alphabet,'length':n,'clues':clues},'truth':solutions[0] if len(solutions)==1 else 'UNKNOWN',
        'solutions':solutions,'possible_count':len(solutions),'allowed_answers':candidates+['UNKNOWN']}
