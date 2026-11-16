from dataclasses import asdict

from selftrap import concept_improve as study
from selftrap.planning_lab import make_world


def test_training_examples_never_use_confirmation_layouts():
    panels = [{"split": "train", "seed": seed, "world": asdict(make_world("numberlink", seed, 3, 3))}
              for seed in (11, 17)]
    panels.append({"split": "confirmation", "seed": 101, "world": {"must_not_be_read": True}})
    examples, sources = study.examples("numberlink", 'def generate(batch):\n return [True for obs in batch]', panels)
    assert len(sources) == 2 and [s["seed"] for s in sources] == [11, 17]
    assert len(examples) == 8 and sum(e["can_finish"] for e in examples) == 4


def test_unsafe_program_is_retained_without_correctness_resampling():
    seen = []
    class Calls:
        def ask(self, model, game, label, prompt):
            seen.append(label)
            return {"model": model.id, "status": "completed", "object": {
                "code": 'def generate(batch):\n return [False for obs in batch]', "p_sound": 1,
                "mechanism": "Synthetic unsafe fixture, never scientific evidence"}}
    task = {"baseline": 'def generate(batch):\n return [True for obs in batch]',
            "examples": [{"observation": {}, "can_finish": True}, {"observation": {}, "can_finish": False}]}
    result = study.author(Calls(), study.MODELS[0], "numberlink", task)
    assert result["construction_valid"] and result["training"]["unsafe"] == 1
    assert seen == ["improve"]
