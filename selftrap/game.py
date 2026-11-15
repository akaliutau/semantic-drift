"""Stage 1: Sokoban transitions. No model, oracle, or research metadata here."""

from dataclasses import dataclass
from hashlib import sha256
import json
import random

Pos = tuple[int, int]
DIRECTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


@dataclass(frozen=True, order=True)
class Push:
    box: Pos
    direction: str

    def to_dict(self) -> dict:
        return {"box": list(self.box), "direction": self.direction}

    @classmethod
    def from_dict(cls, value: dict) -> "Push":
        if not isinstance(value, dict) or set(value) != {"box", "direction"}:
            raise ValueError("A push requires exactly box and direction")
        box = value["box"]
        if not isinstance(box, list) or len(box) != 2 or any(type(x) is not int for x in box):
            raise ValueError("box must contain two integer coordinates")
        if not isinstance(value["direction"], str) or value["direction"] not in DIRECTIONS:
            raise ValueError("Unknown push direction")
        return cls(tuple(box), value["direction"])


def parse_plan(value: object, limit: int = 40) -> list[Push]:
    if not isinstance(value, list) or len(value) > limit:
        raise ValueError(f"Plan must be an array of at most {limit} pushes")
    return [Push.from_dict(item) for item in value]


@dataclass(frozen=True)
class State:
    player: Pos
    boxes: frozenset[Pos]


@dataclass(frozen=True)
class Board:
    height: int
    width: int
    walls: frozenset[Pos]
    goals: frozenset[Pos]

    def floor(self, pos: Pos) -> bool:
        return 0 <= pos[0] < self.height and 0 <= pos[1] < self.width and pos not in self.walls

    def solved(self, state: State) -> bool:
        return state.boxes == self.goals

    def reachable(self, state: State) -> set[Pos]:
        found = {state.player}
        todo = [state.player]
        for row, col in todo:
            for dr, dc in DIRECTIONS.values():
                pos = (row + dr, col + dc)
                if self.floor(pos) and pos not in state.boxes and pos not in found:
                    found.add(pos)
                    todo.append(pos)
        return found

    def push(self, state: State, action: Push) -> State:
        if action.direction not in DIRECTIONS or action.box not in state.boxes:
            raise ValueError("No box at the specified location or unknown direction")
        dr, dc = DIRECTIONS[action.direction]
        row, col = action.box
        behind, dest = (row - dr, col - dc), (row + dr, col + dc)
        if not self.floor(dest) or dest in state.boxes:
            raise ValueError("Box destination is blocked")
        if behind not in self.reachable(state):
            raise ValueError("Player cannot reach the square behind the box")
        return State(action.box, (state.boxes - {action.box}) | {dest})

    def legal_pushes(self, state: State) -> list[Push]:
        reachable = self.reachable(state)
        result = []
        for row, col in sorted(state.boxes):
            for name, (dr, dc) in DIRECTIONS.items():
                dest = (row + dr, col + dc)
                if (row - dr, col - dc) in reachable and self.floor(dest) and dest not in state.boxes:
                    result.append(Push((row, col), name))
        return result

    def render(self, state: State) -> list[str]:
        rows = []
        for row in range(self.height):
            chars = []
            for col in range(self.width):
                pos = row, col
                tile = "#" if pos in self.walls else "." if pos in self.goals else "_"
                if pos in state.boxes:
                    tile = "*" if pos in self.goals else "$"
                if pos == state.player:
                    tile = "+" if pos in self.goals else "@"
                chars.append(tile)
            rows.append("".join(chars))
        return rows


@dataclass(frozen=True)
class Level:
    board: Board
    state: State

    @classmethod
    def from_rows(cls, rows: list[str]) -> "Level":
        if not isinstance(rows, list) or len(rows) < 3 or any(not isinstance(r, str) for r in rows):
            raise ValueError("Board must be an array of strings")
        width = len(rows[0])
        if width < 3 or any(len(row) != width for row in rows):
            raise ValueError("Board must be rectangular")
        walls, goals, boxes, players = set(), set(), set(), []
        for row, line in enumerate(rows):
            for col, tile in enumerate(line):
                pos = row, col
                if tile not in "#_.$*@+":
                    raise ValueError(f"Unknown tile {tile!r}")
                if tile == "#": walls.add(pos)
                if tile in ".*+": goals.add(pos)
                if tile in "$*": boxes.add(pos)
                if tile in "@+": players.append(pos)
                if row in (0, len(rows)-1) or col in (0, width-1):
                    if tile != "#": raise ValueError("Perimeter must be walls")
        if len(players) != 1 or not boxes or len(boxes) != len(goals):
            raise ValueError("Require one player and equal nonzero box/goal counts")
        return cls(Board(len(rows), width, frozenset(walls), frozenset(goals)),
                   State(players[0], frozenset(boxes)))

    def public(self, state: State | None = None, action_seed: int = 0) -> dict:
        state = self.state if state is None else state
        pushes = [p.to_dict() for p in self.board.legal_pushes(state)]
        random.Random(action_seed).shuffle(pushes)
        return {"board": self.board.render(state), "player": list(state.player),
                "boxes": [list(p) for p in sorted(state.boxes)],
                "goals": [list(p) for p in sorted(self.board.goals)],
                "legal_initial_pushes": pushes}

    def replay(self, plan: list[Push], action_seed: int = 0) -> dict:
        state = self.state
        error = None
        executed = 0
        for index, action in enumerate(plan):
            try:
                state = self.board.push(state, action)
                executed += 1
            except ValueError as exc:
                error = {"index": index, "reason": str(exc)}
                break
        public = self.public(state, action_seed)
        public["legal_pushes"] = public.pop("legal_initial_pushes")
        return {**public, "completed": error is None and self.board.solved(state),
                "illegal_action": error, "executed_pushes": executed}


def transform_pos(pos: Pos, size: int, symmetry: int) -> Pos:
    row, col = pos
    if symmetry >= 4: col = size - 1 - col
    for _ in range(symmetry % 4): row, col = col, size - 1 - row
    return row, col


def transform_push(push: Push, size: int, symmetry: int) -> Push:
    start = transform_pos(push.box, size, symmetry)
    dr, dc = DIRECTIONS[push.direction]
    end = transform_pos((push.box[0]+dr, push.box[1]+dc), size, symmetry)
    delta = (end[0]-start[0], end[1]-start[1])
    return Push(start, next(k for k, v in DIRECTIONS.items() if v == delta))


def transform_level(level: Level, symmetry: int) -> Level:
    size = level.board.height
    if size != level.board.width or symmetry not in range(8):
        raise ValueError("Symmetries require a square board and index 0..7")
    tr = lambda p: transform_pos(p, size, symmetry)
    return Level(Board(size, size, frozenset(map(tr, level.board.walls)),
                       frozenset(map(tr, level.board.goals))),
                 State(tr(level.state.player), frozenset(map(tr, level.state.boxes))))


def canonical_id(level: Level) -> str:
    """Ignore box ordering, symmetries and equivalent walking-only positions."""
    variants = []
    for symmetry in range(8):
        other = transform_level(level, symmetry)
        anchor = min(other.board.reachable(other.state))
        rows = other.board.render(State(anchor, other.state.boxes))
        variants.append(json.dumps(rows, separators=(",", ":")))
    return sha256(min(variants).encode()).hexdigest()
