"""Tiny compare-and-swap programs with exhaustive correctness certificates.

A step (i,j) replaces slots i,j by min,max, respectively, even when i > j.
Thresholding commutes with every min/max operation. Therefore any failure to
sort real numbers induces a failure on a 0/1 input: checking all 2**n binary
inputs certifies this fixed, data-independent comparison network. Tests also
check all permutations of four distinct numbers independently.
"""

from itertools import product
import random

from .storage import digest


RULES = """There are four slots, indexed 0,1,2,3, holding arbitrary integers.
Execute the listed steps in order. For step [i,j], if the number in slot i is
GREATER THAN the number in slot j, SWAP those two numbers; otherwise do nothing.
The order [i,j] matters: [3,0] puts the smaller number in slot 3, not slot 0.
A program is VALID if its FINAL array is nondecreasing for EVERY input array.
Intermediate arrays need not be sorted. No other steps or hidden mechanics.
"""

CANONICAL = ((0, 1), (2, 3), (0, 2), (1, 3), (1, 2))


def execute(steps, values):
    values = list(values)
    if len(values) != 4 or len(steps) > 18:
        raise ValueError("Use four slots and at most 18 steps")
    for step in steps:
        if (len(step) != 2 or any(type(i) is not int or not 0 <= i < 4 for i in step)
                or step[0] == step[1]):
            raise ValueError("Every step needs two distinct slot indices in 0..3")
        i, j = step
        if values[i] > values[j]:
            values[i], values[j] = values[j], values[i]
    return values


def certify(steps):
    rows = []
    witness = None
    for values in product((0, 1), repeat=4):
        output = execute(steps, values)
        rows.append({"input": list(values), "output": output})
        if witness is None and output != sorted(values):
            witness = rows[-1]
    return {"valid": witness is None, "binary_inputs_checked": 16,
            "rows": rows, "counterexample": witness,
            "threshold_argument": "Every step uses min and max, which commute with thresholding. An inverted final pair on integer input would remain inverted for a threshold between those two values, producing a binary counterexample."}


def public(steps):
    steps = [list(step) for step in steps]
    return {"id": digest(steps)[:10], "steps": steps}


def mutate(steps, rng):
    child = list(steps)
    operation = rng.choice(("insert", "delete", "move", "replace"))
    if operation == "insert" and len(child) < 12:
        i, j = rng.sample(range(4), 2)
        child.insert(rng.randrange(len(child) + 1), (i, j))
    elif operation == "delete" and len(child) > 4:
        child.pop(rng.randrange(len(child)))
    elif operation == "move":
        child.insert(rng.randrange(len(child)), child.pop(rng.randrange(len(child))))
    else:
        child[rng.randrange(len(child))] = tuple(rng.sample(range(4), 2))
    return tuple(child)


def pool(seed, valid_count=12, invalid_count=12, exclude=()):
    """A bounded evolutionary walk retaining only certified-correct parents."""
    rng = random.Random(seed)
    parents = [CANONICAL]
    rows = {}
    excluded = set(exclude)
    proposals = 0
    while proposals < 1000:
        proposals += 1
        steps = mutate(rng.choice(parents), rng)
        case = public(steps)
        if case["id"] in rows or case["id"] in excluded:
            continue
        certificate = certify(steps)
        valid = certificate["valid"]
        if valid and steps not in parents:
            parents.append(steps)
        wanted = valid_count if valid else invalid_count
        if sum(r["certificate"]["valid"] == valid for r in rows.values()) < wanted:
            rows[case["id"]] = {"public": case, "certificate": certificate}
        if (sum(r["certificate"]["valid"] for r in rows.values()) == valid_count
                and sum(not r["certificate"]["valid"] for r in rows.values()) == invalid_count):
            return {"proposals": proposals, "cases": list(rows.values())}
    raise RuntimeError("The small candidate-pool construction did not finish")


def discovery_panel():
    candidates = pool(41, 20, 20)
    valid = [r for r in candidates["cases"] if r["certificate"]["valid"]]
    invalid = [r for r in candidates["cases"] if not r["certificate"]["valid"]]
    # A backward comparison is not itself a bug: later steps may repair it.
    nontrivial = [r for r in valid if tuple(map(tuple, r["public"]["steps"][-5:])) != CANONICAL
                  and any(i > j for i, j in r["public"]["steps"])]
    if not nontrivial:
        raise RuntimeError("No unusual valid candidate without a canonical suffix")
    unusual = max(nontrivial, key=lambda r: (sum(i > j for i, j in r["public"]["steps"]), len(r["public"]["steps"])))
    sparse = min(invalid, key=lambda r: (sum(row["output"] != sorted(row["input"]) for row in r["certificate"]["rows"]), -len(r["public"]["steps"])))
    canonical = {"public": public(CANONICAL), "certificate": certify(CANONICAL)}
    broken_steps = CANONICAL[:-1]
    broken = {"public": public(broken_steps), "certificate": certify(broken_steps)}
    rows = [canonical, unusual, sparse, broken]
    if len({r["public"]["id"] for r in rows}) != 4:
        raise RuntimeError("Discovery cases must be distinct")
    random.Random(73).shuffle(rows)
    return {"proposals": candidates["proposals"], "cases": rows}


def render(case, path):
    """One small exact program, readable without code or a visual model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = case["public"]["steps"]
    fig, ax = plt.subplots(figsize=(max(7, len(steps) * .65), 3.7))
    for wire in range(4):
        ax.plot((-.6, len(steps) - .4), (3 - wire, 3 - wire), color="#c6ccd7", lw=2)
        ax.text(-.8, 3 - wire, str(wire), ha="right", va="center", fontsize=12)
    for x, (i, j) in enumerate(steps):
        ax.annotate("", xy=(x, 3-j), xytext=(x, 3-i),
                    arrowprops={"arrowstyle": "-|>", "color": "#c15b15" if i > j else "#345591", "lw": 2})
        ax.plot(x, 3-i, "o", color="#345591", markersize=5)
        ax.text(x, -.55, str(x + 1), ha="center", fontsize=9)
    verdict = "correct for every input" if case["certificate"]["valid"] else "has a counterexample"
    ax.set_title(f"A four-slot sorting program: {verdict}", loc="left", fontsize=13, weight="bold", pad=18)
    fig.text(.5, .025, "Follow steps left to right. Each arrow moves the larger value to its tip.\nAn upward arrow may be unusual without making the whole program wrong.", ha="center", fontsize=10)
    ax.set_xlim(-1, len(steps))
    ax.set_ylim(-.8, 3.4)
    ax.axis("off")
    fig.tight_layout(rect=(0, .12, 1, 1))
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
