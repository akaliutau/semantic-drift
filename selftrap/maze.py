"""A three-room laboratory for discovering insufficient graph representations.

Only one extra mechanic: entering the key room permanently acquires a reusable
key. One undirected door requires it. The simulator supplies transitions; the
model supplies only the expression identifying a graph node. This deliberately
isolates state abstraction from perception, rule extraction, and search.
"""

import ast
from collections import deque
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path


RULES = (
    "Rooms are connected by undirected passages. Moving through a passage takes "
    "one step. Entering the key room automatically gives you one reusable key. "
    "You keep it forever. The one locked passage can be crossed in either "
    "direction only while carrying that key. Reaching the exit ends the game. "
    "There are no other mechanics. Find a shortest valid route to the exit."
)


@dataclass(frozen=True)
class State:
    room: str
    has_key: bool = False


@dataclass(frozen=True)
class Maze:
    rooms: tuple[str, ...]
    passages: tuple[tuple[str, str], ...]
    start: str
    goal: str
    key_room: str | None = None
    locked_passage: tuple[str, str] | None = None

    def __post_init__(self):
        if len(set(self.rooms)) != len(self.rooms) or not 2 <= len(self.rooms) <= 12:
            raise ValueError("Use 2..12 distinct rooms")
        if self.start not in self.rooms or self.goal not in self.rooms or self.start == self.goal:
            raise ValueError("Distinct start and goal must be rooms")
        edges = [frozenset(e) for e in self.passages]
        if len(set(edges)) != len(edges) or any(len(e) != 2 or not e <= set(self.rooms) for e in edges):
            raise ValueError("Passages must be distinct undirected room pairs")
        if self.key_room is not None and self.key_room not in self.rooms:
            raise ValueError("Unknown key room")
        if self.locked_passage is not None and frozenset(self.locked_passage) not in edges:
            raise ValueError("The locked passage must exist")

    @property
    def initial(self):
        return State(self.start, self.start == self.key_room)

    def moves(self, state):
        if state.room not in self.rooms or type(state.has_key) is not bool:
            raise ValueError("Invalid state")
        if state.room == self.goal:
            return {}
        result = {}
        for other in self.rooms:
            edge = frozenset((state.room, other))
            if edge not in [frozenset(e) for e in self.passages]:
                continue
            if self.locked_passage is not None and edge == frozenset(self.locked_passage) and not state.has_key:
                continue
            result[other] = State(other, state.has_key or other == self.key_room)
        return result

    def replay(self, path):
        state = self.initial
        for destination in path:
            options = self.moves(state)
            if destination not in options:
                raise ValueError(f"Illegal move from {state.room} to {destination}")
            state = options[destination]
        return state

    def public(self):
        return {**asdict(self), "rules": RULES}


def encoder(expression):
    """Compile a bounded, side-effect-free expression over the two observations.

    No calls, attributes, imports, indexing, arithmetic, or arbitrary evaluation
    objects are admitted. The only runtime values are a room string and a bool.
    """
    if not isinstance(expression, str) or len(expression) > 500:
        raise ValueError("node_expression must be a string of at most 500 characters")
    tree = ast.parse(expression, mode="eval")
    allowed = (ast.Expression, ast.Name, ast.Load, ast.Tuple, ast.Constant,
               ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not,
               ast.Compare, ast.Eq, ast.NotEq, ast.IfExp)
    nodes = list(ast.walk(tree))
    if len(nodes) > 120 or any(not isinstance(n, allowed) for n in nodes):
        raise ValueError("Use names, tuples, constants, comparisons, and Boolean expressions only")
    if any(isinstance(n, ast.Name) and n.id not in {"room", "has_key"} for n in nodes):
        raise ValueError("Only room and has_key are available")
    if any(isinstance(n, ast.Constant) and type(n.value) not in {str, bool, int, type(None)} for n in nodes):
        raise ValueError("Unsupported constant")
    code = compile(tree, "<node_expression>", "eval")

    def encode(state):
        value = eval(code, {"__builtins__": {}}, {"room": state.room, "has_key": state.has_key})
        hash(value)
        return value

    return encode


def reachable(maze):
    """Enumerate actual reachable states, retaining a valid path to each."""
    paths = {maze.initial: []}
    queue = deque([maze.initial])
    while queue:
        state = queue.popleft()
        for action, successor in maze.moves(state).items():
            if successor not in paths:
                paths[successor] = paths[state] + [action]
                queue.append(successor)
    return paths


def plan(maze, encode):
    """The SAME BFS in every arm; only the graph-node identity changes."""
    queue = deque([(maze.initial, [])])
    visited = {encode(maze.initial)}
    while queue:
        state, path = queue.popleft()
        if state.room == maze.goal:
            return path
        for action, successor in maze.moves(state).items():
            node = encode(successor)
            if node not in visited:
                visited.add(node)
                queue.append((successor, path + [action]))
    return None


def _difference(maze, first, second, encode):
    if (first.room == maze.goal) != (second.room == maze.goal):
        return {"kind": "goal_mismatch", "destination": None}
    left, right = maze.moves(first), maze.moves(second)
    for action in maze.rooms:
        if (action in left) != (action in right):
            return {"kind": "legal_move_mismatch", "destination": action,
                    "legal_from_first": action in left, "legal_from_second": action in right}
        if action in left and encode(left[action]) != encode(right[action]):
            return {"kind": "successor_mismatch", "destination": action}
    return None


def collision(maze, encode):
    """A witness that a node merges states with different transition semantics.

    Equality of goal labels and of action-indexed successor classes is the
    finite deterministic bisimulation condition checked here. This is stronger
    than merely testing one successful route; it is not a universal claim about
    every policy that might use the textual problem in some other way.
    """
    paths = reachable(maze)
    buckets = {}
    for state in paths:
        key = encode(state)
        if key in buckets:
            first = buckets[key]
            difference = _difference(maze, first, state, encode)
            if difference:
                return {"first": asdict(first), "second": asdict(state),
                        "path_to_first": paths[first], "path_to_second": paths[state],
                        "difference": difference}
        else:
            buckets[key] = state
    return None


def verify_certificate(maze, expression, certificate):
    """Replay both witness histories and recheck the claimed semantic difference."""
    first = maze.replay(certificate["path_to_first"])
    second = maze.replay(certificate["path_to_second"])
    encode = encoder(expression)
    return (asdict(first) == certificate["first"]
            and asdict(second) == certificate["second"]
            and encode(first) == encode(second)
            and _difference(maze, first, second, encode) == certificate["difference"]
            and certificate["difference"] is not None)


def evaluate(maze, expression):
    encode = encoder(expression)
    truth = plan(maze, lambda s: (s.room, s.has_key))
    candidate = plan(maze, encode)
    witness = collision(maze, encode)
    valid_plan = candidate is not None and maze.replay(candidate).room == maze.goal
    return {"solvable": truth is not None, "optimal_steps": None if truth is None else len(truth),
            "plan": candidate, "solved": valid_plan,
            "optimal": valid_plan and len(candidate) == len(truth),
            "sufficient_on_reachable_states": witness is None, "certificate": witness,
            "reachable_states": len(reachable(maze))}


def discover():
    """Ordered mechanical search, not model-generated physics or large testing.

    Search starts at 3 rooms, with the fewest passages first. The key is in an
    internal room and exactly one existing passage is locked. Stop at the first
    solvable world where room-only BFS fails and an aliasing witness exists.
    Minimality applies only to this explicitly bounded construction grammar.
    """
    considered = 0
    for count in (3, 4):
        rooms = tuple(chr(65 + i) for i in range(count))
        edges = list(combinations(rooms, 2))
        for size in range(count - 1, len(edges) + 1):
            for passages in combinations(edges, size):
                for key_room in rooms[1:-1]:
                    for door in passages:
                        maze = Maze(rooms, passages, rooms[0], rooms[-1], key_room, door)
                        considered += 1
                        result = evaluate(maze, "room")
                        if result["solvable"] and not result["solved"] and result["certificate"]:
                            assert verify_certificate(maze, "room", result["certificate"])
                            return {"world": maze, "considered": considered, "evaluation": result}
    raise RuntimeError("No representation counterexample in the bounded grammar")


def held_out():
    """Four fixed, prospectively sealed transfer cases; no new mechanics."""
    return [
        Maze(("entrance", "alcove", "exit"),
             (("entrance", "alcove"), ("entrance", "exit")),
             "entrance", "exit", "alcove", ("entrance", "exit")),
        Maze(("P", "Q", "R", "T", "U"),
             (("P", "Q"), ("Q", "R"), ("Q", "T"), ("T", "U")),
             "P", "U", "R", ("T", "U")),
        Maze(("north", "west", "east", "south"),
             (("north", "west"), ("west", "east"), ("east", "north"),
              ("north", "south"), ("east", "south")),
             "north", "south", "east", ("north", "south")),
        Maze(("one", "two", "three", "four"),
             (("one", "two"), ("two", "three"), ("three", "four")),
             "one", "four"),
    ]


def render(maze, path):
    """Human-facing diagram. Models receive the exact symbolic room list."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch

    if len(maze.rooms) != 3:
        raise ValueError("This diagram is for the minimal three-room example")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), facecolor="white")
    left, right = axes
    positions = {maze.start: (0.45, 0.48), maze.key_room: (0.45, 0.91), maze.goal: (1.45, 0.48)}
    for a, b in maze.passages:
        x, y = positions[a], positions[b]
        locked = maze.locked_passage is not None and frozenset((a, b)) == frozenset(maze.locked_passage)
        left.plot([x[0], y[0]], [x[1], y[1]], color="#cc6b21" if locked else "#667085",
                  linewidth=3, linestyle="--" if locked else "-", zorder=1)
        if locked:
            left.text(0.94, 0.56, "locked door", ha="center", color="#a34b13", fontsize=11)
    for room, (x, y) in positions.items():
        left.add_patch(Circle((x, y), 0.115, facecolor="#e8f0ff", edgecolor="#345591", linewidth=2, zorder=2))
        name = "start" if room == maze.start else "exit" if room == maze.goal else "key"
        left.text(x, y, f"{room}\n{name}", ha="center", va="center", fontsize=11, zorder=3)
    left.text(0.98, 0.19, "Get the key, return to the start, then exit.", ha="center", fontsize=11)
    left.set_title("The whole game: one reusable key", loc="left", fontsize=13, weight="bold", pad=18)
    left.set_xlim(0.1, 1.8)
    left.set_ylim(0.05, 1.14)

    nodes = [(maze.start, False), (maze.key_room, True), (maze.start, True), (maze.goal, True)]
    for i, (room, key) in enumerate(nodes):
        x = 0.4 + i * 0.86
        color = "#fff0b8" if key else "#e8f0ff"
        right.add_patch(Circle((x, 0.7), 0.27, facecolor=color, edgecolor="#345591", linewidth=2, zorder=2))
        right.text(x, 0.7, f"{room}\n{'key' if key else 'no key'}", ha="center", va="center", fontsize=10)
        if i:
            right.add_patch(FancyArrowPatch((x - 0.59, 0.7), (x - 0.28, 0.7),
                                            arrowstyle="-|>", mutation_scale=15, color="#667085"))
    right.annotate("Same room, different legal moves", xy=(0.4, 0.34), xytext=(1.26, -0.05),
                   ha="center", fontsize=10, arrowprops={"arrowstyle": "-", "color": "#b42318"})
    right.annotate("", xy=(2.12, 0.34), xytext=(1.26, 0.01),
                   arrowprops={"arrowstyle": "-", "color": "#b42318"})
    right.text(1.68, -0.43, "A room-only map merges the two start states.\nAdding the key bit preserves the distinction.",
               ha="center", va="center", fontsize=11)
    right.set_title("The missing distinction", loc="left", fontsize=13, weight="bold", pad=18)
    right.set_xlim(0, 3.4)
    right.set_ylim(-0.75, 1.3)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout(pad=2)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
