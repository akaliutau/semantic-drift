"""Three small deterministic games and exact reachable-state certificates.

Cell r*width+c denotes row r, column c. Walls block undirected adjacent edges.
All observations are structured; PNGs are for human inspection, not perception
tests. Ricochet is an explicitly simplified, solo sliding-robot adaptation.
"""

from collections import deque
from dataclasses import asdict, dataclass
from functools import lru_cache
import random

from .storage import digest


DIRECTIONS = (("U", -1, 0), ("R", 0, 1), ("D", 1, 0), ("L", 0, -1))
RULES = {
    "dots": "Draw one orthogonal path from number 1 through all numbered cells in increasing order. Never revisit a cell or cross a wall. Cover EVERY cell, finishing at the highest number. Entering the last number early leaves no completion.",
    "ricochet": "Solo adaptation with three robots. Robot 0 must land on the target. Each action slides any one robot U/R/D/L until the next step would hit an edge, wall or other robot. It cannot stop earlier. Helper moves are allowed and count as actions. The game ends when robot 0 reaches the target. No diagonal mirrors or mandatory-ricochet rule.",
    "numberlink": "Join each numbered pair by orthogonal, vertex-disjoint simple paths. Unused cells ARE ALLOWED; filling the grid is NOT required. Solve pairs in their listed order, growing the current path from its first endpoint. Other numbered endpoints cannot be crossed. Reaching its second endpoint starts the next pair automatically. This fixed drawing order preserves every possible complete set of disjoint paths.",
}


@dataclass(frozen=True)
class World:
    game: str
    width: int
    height: int
    walls: tuple = ()
    points: tuple = ()
    pairs: tuple = ()
    robots: tuple = ()
    target: int = -1

    def __post_init__(self):
        n = self.width * self.height
        if self.game not in RULES or not 2 <= self.width <= 6 or not 2 <= self.height <= 6:
            raise ValueError("Unknown game or board outside 2..6 per side")
        edges = [tuple(sorted(e)) for e in self.walls]
        if (len(set(edges)) != len(edges) or any(len(e) != 2 or
                any(type(v) is not int or not 0 <= v < n for v in e) or
                abs(e[0] // self.width - e[1] // self.width) + abs(e[0] % self.width - e[1] % self.width) != 1 for e in edges)):
            raise ValueError("Walls must be distinct adjacent cell pairs")
        cells = self.points if self.game == "dots" else tuple(v for p in self.pairs for v in p) if self.game == "numberlink" else self.robots
        if len(cells) != len(set(cells)) or any(type(v) is not int or not 0 <= v < n for v in cells):
            raise ValueError("Game locations must be distinct cells")
        if self.game == "dots" and len(self.points) < 2:
            raise ValueError("At least two ordered dots")
        if self.game == "numberlink" and (not self.pairs or any(len(p) != 2 for p in self.pairs) or self.walls):
            raise ValueError("Use nonempty pairs and standard wall-free Numberlink")
        if self.game == "ricochet" and (len(self.robots) != 3 or not 0 <= self.target < n):
            raise ValueError("Use three robots and an in-board target")

    @property
    def initial(self):
        if self.game == "ricochet":
            return self.robots
        first = self.points[0] if self.game == "dots" else self.pairs[0][0]
        return (first, 1 << first, 1 if self.game == "dots" else 0)

    def adjacent(self, cell):
        row, col = divmod(cell, self.width)
        blocked = {frozenset(e) for e in self.walls}
        for direction, dr, dc in DIRECTIONS:
            rr, cc = row + dr, col + dc
            if 0 <= rr < self.height and 0 <= cc < self.width:
                other = rr * self.width + cc
                if frozenset((cell, other)) not in blocked:
                    yield direction, other

    def goal(self, state):
        if self.game == "ricochet":
            return state[0] == self.target
        head, mask, progress = state
        if self.game == "dots":
            return progress == len(self.points) and head == self.points[-1] and mask == (1 << (self.width * self.height)) - 1
        return progress == len(self.pairs)

    def moves(self, state):
        if self.goal(state):
            return {}
        if self.game == "ricochet":
            result = {}
            for robot, position in enumerate(state):
                for direction, _, _ in DIRECTIONS:
                    current = position
                    for _ in range(max(self.width, self.height)):
                        options = dict(self.adjacent(current))
                        if direction not in options or options[direction] in state:
                            break
                        current = options[direction]
                    if current != position:
                        child = list(state)
                        child[robot] = current
                        result[f"{robot}:{direction}"] = tuple(child)
            return result
        head, mask, progress = state
        if self.game == "dots" and progress == len(self.points):
            return {}
        result = {}
        for action, cell in self.adjacent(head):
            if mask & (1 << cell):
                continue
            next_progress = progress
            next_head, next_mask = cell, mask | (1 << cell)
            if self.game == "dots":
                if cell in self.points:
                    if cell != self.points[progress]:
                        continue
                    next_progress += 1
            else:
                terminal_cells = {v for pair in self.pairs for v in pair}
                if cell in terminal_cells and cell != self.pairs[progress][1]:
                    continue
                if cell == self.pairs[progress][1]:
                    next_progress += 1
                    if next_progress < len(self.pairs):
                        next_head = self.pairs[next_progress][0]
                        next_mask |= 1 << next_head
            result[action] = (next_head, next_mask, next_progress)
        return result

    def public(self):
        data = {"game": self.game, "width": self.width, "height": self.height, "walls": [list(e) for e in self.walls]}
        if self.game == "dots":
            data["numbers"] = list(self.points)
        elif self.game == "ricochet":
            data["target"] = self.target
        else:
            data["pairs"] = [list(p) for p in self.pairs]
        return data

    def observe(self, state):
        if self.game == "ricochet":
            visible = {"robots": list(state)}
        else:
            head, mask, progress = state
            visible = {"head": head, "used": [i for i in range(self.width * self.height) if mask & (1 << i)]}
            if self.game == "dots":
                visible["next_number"] = progress + 1 if progress < len(self.points) else None
            else:
                visible["pair_index"] = progress
        return {"board": self.public(), "state": visible}

    def replay(self, path, start=None):
        state = self.initial if start is None else tuple(start)
        for action in path:
            choices = self.moves(state)
            if action not in choices:
                raise ValueError(f"Illegal action {action}")
            state = choices[action]
        return state


class Arena:
    def __init__(self, world, max_states=60000):
        self.world, self.parent, self.edges = world, {world.initial: None}, {}
        self.reverse = {}
        queue = deque([world.initial])
        while queue:
            state = queue.popleft()
            children = world.moves(state)
            self.edges[state] = children
            for action, child in children.items():
                self.reverse.setdefault(child, []).append(state)
                if child not in self.parent:
                    if len(self.parent) >= max_states:
                        raise ValueError("Exact arena exceeds the frozen state cap")
                    self.parent[child] = (state, action)
                    queue.append(child)
        self.distance = {s: 0 for s in self.parent if world.goal(s)}
        queue = deque(self.distance)
        while queue:
            state = queue.popleft()
            for previous in self.reverse.get(state, []):
                if previous not in self.distance:
                    self.distance[previous] = self.distance[state] + 1
                    queue.append(previous)

    def history(self, state):
        actions = []
        while self.parent[state] is not None:
            state, action = self.parent[state]
            actions.append(action)
        return list(reversed(actions))

    def completion(self, state):
        if state not in self.distance:
            return None
        actions = []
        while self.distance[state]:
            action, child = next((a, c) for a, c in self.edges[state].items()
                                 if self.distance.get(c) == self.distance[state] - 1)
            actions.append(action)
            state = child
        return actions

    def certificate(self, state, move_budget=None):
        path = self.history(state)
        completion = self.completion(state)
        if self.world.replay(path) != state:
            raise AssertionError("Reachability certificate failed")
        if completion is not None and not self.world.goal(self.world.replay(completion, state)):
            raise AssertionError("Completion certificate failed")
        possible = completion is not None and (move_budget is None or len(completion) <= move_budget)
        observation = self.world.observe(state)
        if move_budget is not None:
            observation["moves_left"] = move_budget
        return {"world": asdict(self.world), "observation": observation,
                "history": path, "can_finish": possible,
                "completion": completion if possible else None,
                "shortest_remaining": None if completion is None else len(completion),
                "reachable_states_checked": len(self.parent)}


def _hamiltonian(width, height, rng):
    n = width * height
    start = rng.choice([i for i in range(n) if n % 2 == 0 or (i // width + i % width) % 2 == 0])
    budget = [0]
    def visit(path, used):
        budget[0] += 1
        if len(path) == n:
            return path
        if budget[0] > 100000:
            return None
        row, col = divmod(path[-1], width)
        options = [(row + dr) * width + col + dc for _, dr, dc in DIRECTIONS
                   if 0 <= row + dr < height and 0 <= col + dc < width and (row + dr) * width + col + dc not in used]
        rng.shuffle(options)
        for cell in options:
            result = visit(path + [cell], used | {cell})
            if result is not None:
                return result
        return None
    result = visit([start], {start})
    if result is None:
        raise ValueError("Small Hamiltonian construction cap")
    return result


@lru_cache(maxsize=64)
def make_world(game, seed, width=4, height=4):
    rng = random.Random(seed)
    n = width * height
    if game in {"dots", "numberlink"}:
        path = _hamiltonian(width, height, rng)
        if game == "numberlink":
            # Three disjoint witness paths, separated by unused cells.
            cuts = [0, n // 3, 2 * n // 3, n]
            pairs = tuple((path[cuts[i]], path[cuts[i+1] - 2]) for i in range(3))
            return World(game, width, height, pairs=pairs)
        path_edges = {frozenset(e) for e in zip(path, path[1:])}
        walls = []
        blank = World("ricochet", width, height, robots=(0, 1, 2), target=n-1)
        for cell in range(n):
            for _, other in blank.adjacent(cell):
                if cell < other and frozenset((cell, other)) not in path_edges and rng.random() < .4:
                    walls.append((cell, other))
        points = tuple(path[i] for i in (0, n // 3, 2 * n // 3, n-1))
        return World(game, width, height, walls=tuple(walls), points=points)
    if game != "ricochet":
        raise ValueError("Unknown game")
    for _ in range(30):
        robots = tuple(rng.sample(range(n), 3))
        target = rng.choice([r * width + c for r in range(1, height-1) for c in range(1, width-1)])
        walls = tuple((cell, cell + 1) for cell in range(n - 1)
                      if cell // width == (cell + 1) // width and rng.random() < .08)
        world = World(game, width, height, walls=walls, robots=robots, target=target)
        if world.initial in Arena(world).distance:
            return world
    raise ValueError("No solvable small robot board")


def records(arena, count=32):
    """Depth-spread bounded-horizon probes; goal states are checked before pruning.

    moves_left is an explicit suffix action budget, not an unknown mechanic.
    All games preserve their original rules. Exact shortest distances provide
    labels; a false rejection is certified just by replaying a short solution.
    """
    if count < 4 or count % 2:
        raise ValueError("Use an even probe count >=4")
    states = [s for s in arena.parent if not arena.world.goal(s)]
    states.sort(key=lambda s: (len(arena.history(s)), s))
    if len(states) > 128:
        states = [states[i * (len(states)-1) // 127] for i in range(128)]
    candidates = [(s, budget) for s in states for budget in (1, 2, 3, 5, 8, 16)]
    selected = []
    for winning in (True, False):
        options = [(s,b) for s,b in candidates if (s in arena.distance and arena.distance[s] <= b) == winning]
        if len(options) > count // 2:
            options = [options[i * (len(options)-1) // (count//2 - 1)] for i in range(count//2)]
        selected += options
    result = []
    for state, budget in selected:
        certificate = arena.certificate(state, budget)
        observation = certificate["observation"]
        result.append({"id": digest(observation)[:12], "observation": observation,
                       "can_finish": certificate["can_finish"], "certificate": certificate})
    random.Random(digest(arena.world.public())).shuffle(result)
    return result


def restore(data):
    return World(data["game"], data["width"], data["height"],
                 walls=tuple(tuple(e) for e in data.get("walls", [])),
                 points=tuple(data.get("points", [])), pairs=tuple(tuple(p) for p in data.get("pairs", [])),
                 robots=tuple(data.get("robots", [])), target=data.get("target", -1))


def verify(certificate):
    world = restore(certificate["world"])
    state = world.replay(certificate["history"])
    expected = world.observe(state)
    budget = certificate["observation"].get("moves_left")
    if budget is not None:
        expected["moves_left"] = budget
    if expected != certificate["observation"]:
        return False
    if certificate["can_finish"]:
        route = certificate["completion"]
        return (isinstance(route, list) and (budget is None or len(route) <= budget)
                and world.goal(world.replay(route, state)))
    arena = Arena(world)
    return state not in arena.distance or budget is not None and arena.distance[state] > budget


def render(world, state, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(5, 5))
    w, h = world.width, world.height
    used = set(world.observe(state)["state"].get("used", []))
    for cell in range(w*h):
        row, col = divmod(cell, w)
        ax.add_patch(plt.Rectangle((col, h-1-row), 1, 1, facecolor="#e2e8f0" if cell in used else "white", edgecolor="#cbd5e1"))
    def mark(cell, text, color):
        row, col = divmod(cell, w)
        ax.add_patch(Circle((col+.5, h-row-.5), .27, facecolor=color, edgecolor="#1e293b"))
        ax.text(col+.5, h-row-.5, str(text), ha="center", va="center", weight="bold")
    if world.game == "dots":
        for i, cell in enumerate(world.points):
            mark(cell, i+1, "#bae6fd")
    elif world.game == "numberlink":
        for i, pair in enumerate(world.pairs):
            for cell in pair:
                mark(cell, i+1, ("#fecaca", "#bbf7d0", "#bfdbfe")[i % 3])
    else:
        mark(world.target, "★", "#fde68a")
        for i, cell in enumerate(state):
            mark(cell, i, ("#fca5a5", "#93c5fd", "#86efac")[i])
    for first, second in world.walls:
        r1, c1 = divmod(first, w); r2, c2 = divmod(second, w)
        if r1 == r2:
            x = max(c1, c2); ax.plot((x, x), (h-1-r1, h-r1), color="#111827", lw=4)
        else:
            y = h-max(r1, r2); ax.plot((c1, c1+1), (y, y), color="#111827", lw=4)
    ax.set(xlim=(0,w), ylim=(0,h), aspect="equal", title={"dots":"Ordered dots with walls", "ricochet":"Sliding robots: land red 0 on the star", "numberlink":"Numberlink: join matching pairs"}[world.game])
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)
