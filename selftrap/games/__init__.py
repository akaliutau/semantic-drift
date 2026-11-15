"""Small finite games with exact, independent observation oracles."""
from importlib import import_module

ORDER=('mastermind','blackbox','minesweeper','zendo')

def load_game(name):
    if name not in ORDER:raise ValueError('Unknown game')
    return import_module('selftrap.games.'+name)
