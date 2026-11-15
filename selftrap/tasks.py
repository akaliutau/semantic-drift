"""Public decision task and submission schema; no experimental history."""
from . import switches
from .storage import dumps

def actor_text(c):
    return switches.RULES+'''
Choose the FIRST move of a plan that reaches the all-zero goal within the move
budget. A verified winning plan exists. All five moves are legal. Choose one
index, using choose_move. Next states are already computed, so you can focus
on planning rather than parsing or calculating the first transition.
'''+dumps(switches.public(c))

def move_tool():
    return {'type':'function','name':'choose_move','description':'Commit to the first move.',
        'parameters':{'type':'object','additionalProperties':False,'required':['action'],
            'properties':{'action':{'type':'integer','minimum':0,'maximum':4}}}}
