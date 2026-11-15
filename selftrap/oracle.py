"""Independent integer-cell push BFS. Caps produce UNKNOWN, never UNSOLVABLE.

This implementation deliberately does not call Board.push/reachable/replay.
It independently checks witnesses and exhausts the finite reachable state graph.
"""

from collections import deque
from dataclasses import dataclass
from time import perf_counter

from .game import Level, Push


@dataclass
class SearchResult:
    status: str
    plan: list[Push]
    states: int
    seconds: float

    def to_dict(self) -> dict:
        return {"status": self.status, "plan": [p.to_dict() for p in self.plan],
                "states": self.states, "seconds": self.seconds}


class Oracle:
    def __init__(self, level: Level):
        self.width = level.board.width
        self.floor = {r*self.width+c for r in range(level.board.height)
                      for c in range(self.width) if (r, c) not in level.board.walls}
        self.goals = frozenset(self.encode(p) for p in level.board.goals)
        self.player = self.encode(level.state.player)
        self.boxes = frozenset(self.encode(p) for p in level.state.boxes)
        self.steps = (("up", -self.width), ("down", self.width), ("left", -1), ("right", 1))

    def encode(self, pos: tuple[int, int]) -> int:
        return pos[0]*self.width + pos[1]

    def flood(self, player: int, boxes: frozenset[int]) -> set[int]:
        reached = {player}
        todo = [player]
        for cell in todo:
            for _, delta in self.steps:
                nxt = cell + delta
                if nxt in self.floor and nxt not in boxes and nxt not in reached:
                    reached.add(nxt)
                    todo.append(nxt)
        return reached

    def replay(self, plan: list[Push]) -> bool:
        player, boxes = self.player, self.boxes
        for push in plan:
            box = self.encode(push.box)
            delta = dict(self.steps).get(push.direction)
            if (push.box[1] < 0 or push.box[1] >= self.width or delta is None or box not in boxes
                    or box-delta not in self.flood(player, boxes)
                    or box+delta not in self.floor or box+delta in boxes):
                return False
            boxes = (boxes - {box}) | {box+delta}
            player = box
        return boxes == self.goals

    def search(self, max_states: int = 100_000, timeout: float = 5.0) -> SearchResult:
        if max_states < 1 or timeout <= 0:
            raise ValueError("Oracle caps must be positive")
        start_time = perf_counter()
        start = (min(self.flood(self.player, self.boxes)), tuple(sorted(self.boxes)))
        queue = deque([start])
        parents = {start: None}
        def result(status: str, plan: list[Push] | None = None) -> SearchResult:
            return SearchResult(status, plan or [], len(parents), perf_counter()-start_time)
        while queue:
            if perf_counter()-start_time >= timeout: return result("UNKNOWN")
            key = queue.popleft()
            player, packed_boxes = key
            boxes = frozenset(packed_boxes)
            if boxes == self.goals:
                plan = []
                while parents[key] is not None:
                    key, action = parents[key]
                    plan.append(action)
                return result("SOLVED", list(reversed(plan)))
            reached = self.flood(player, boxes)
            for box in sorted(boxes):
                for direction, delta in self.steps:
                    if box-delta not in reached or box+delta not in self.floor or box+delta in boxes:
                        continue
                    new_boxes = (boxes - {box}) | {box+delta}
                    new_key = (min(self.flood(box, new_boxes)), tuple(sorted(new_boxes)))
                    if new_key in parents: continue
                    if len(parents) >= max_states: return result("UNKNOWN")
                    action = Push(divmod(box, self.width), direction)
                    parents[new_key] = key, action
                    queue.append(new_key)
        return result("UNSOLVABLE")
