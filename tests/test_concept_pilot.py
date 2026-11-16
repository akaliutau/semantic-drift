import pytest

from selftrap import concept_pilot as study
from selftrap.planning_lab import Arena, World, make_world, records, verify
from selftrap.providers import ModelResponse
from selftrap.runtime import RESPONSE_IDS
from selftrap.storage import digest, dumps, read_json, write_json


KEEP_ALL = 'def generate(batch):\n return [True for obs in batch]'


def test_safe_abstaining_filter_has_no_fabricated_acceleration():
    rows = records(Arena(make_world("numberlink", 11, 3, 3)), 8)
    result = study.score(KEEP_ALL, rows)
    assert result["unsafe"] == result["dead_states_pruned"] == 0
    assert result["kept"] == len(rows)
    with pytest.raises(ValueError):
        study.run_filter('def generate(batch):\n return [1 for obs in batch]', [{}])
    with pytest.raises(ValueError):
        study.run_filter('def generate(batch):\n return inspect(batch)', [{}])


def test_overgeneralized_connectivity_pruner_has_a_real_numberlink_counterexample():
    world = World("numberlink", 4, 3, pairs=((4, 7), (8, 11)))
    arena = Arena(world)
    state = world.replay(["R", "R", "R"])
    certificate = arena.certificate(state, 3)
    assert verify(certificate)
    row = {"id": "fixture-only", "observation": certificate["observation"], "can_finish": True}
    code = '''def generate(batch):
    result = []
    for obs in batch:
        w = obs['board']['width']
        h = obs['board']['height']
        used = set(obs['state']['used'])
        head = obs['state']['head']
        free = set(range(w*h)) - used
        free.add(head)
        reached = {head}
        queue = [head]
        for cell in queue:
            for other in free:
                if abs(cell//w-other//w)+abs(cell%w-other%w)==1 and other not in reached:
                    reached.add(other)
                    queue.append(other)
        result.append(len(reached)==len(free))
    return result
'''
    assert study.score(code, [row])["unsafe"] == 1


def test_six_cold_authors_and_resume_are_bounded_without_paid_calls(tmp_path, monkeypatch):
    calls = []
    panels = {}
    for game in study.GAMES:
        rows = records(Arena(make_world(game, 11, 3, 3)), 8)
        panels[game] = [{"split": "train", "seed": 11, "rows": rows},
                        {"split": "discovery", "seed": 41, "rows": rows}]
    design = {"budget_before": None, "panels": panels}

    class Provider:
        def __init__(self, **kwargs):
            pass

        def complete(self, model, messages, **kwargs):
            calls.append((model.id, messages))
            obj = {"code": KEEP_ALL, "mechanism": "Synthetic fixture only", "safety_argument": "No states discarded", "p_sound": 1}
            return ModelResponse(dumps(obj), [], {"status": "completed", "model": RESPONSE_IDS[model.id]},
                                 .01, {"cost": .0001}, .0001, "credits")

        def close(self):
            pass

    monkeypatch.setattr(study, "BudgetProvider", Provider)
    monkeypatch.setattr(study, "prepare", lambda *args: design)
    monkeypatch.setattr(study, "usage", lambda: {"calls": len(calls), "reported_credits": len(calls)*.0001,
        "uncertain_reserve": 0, "input_tokens": 0, "output_tokens": 0, "in_flight": 0, "hard_ceiling": 190})
    study.run(tmp_path, live=True)
    assert len(calls) == 6
    for game in study.GAMES:
        r = read_json(tmp_path / f"{game}-cold.json")
        assert len(r["authors"]) == 2
        assert all(a["construction_valid"] and a["discovery"][0]["unsafe"] == 0 for a in r["authors"])
    study.run(tmp_path, live=True)
    assert len(calls) == 6


def test_completed_authoring_reuse_requires_identical_prompt_and_receipt(tmp_path):
    previous = tmp_path / "prior.json"
    row = {"status": "completed", "prompt_hash": digest("same"), "object": {"code": KEEP_ALL}}
    write_json(previous, row)
    design = {"spec": {"reused_authorings": [{"game": "dots", "model": study.MODELS_PINNED[1].id,
               "label": "cold", "path": str(previous), "hash": digest(row)}]}}
    calls = study.Calls(tmp_path / "new", design, None)
    model = study.MODELS_PINNED[1]
    result = calls.ask(model, "dots", "cold", "same")
    assert result["reused_from"] == str(previous)
    assert calls.count == 0
    with pytest.raises(ValueError):
        calls.ask(model, "dots", "cold", "changed")
