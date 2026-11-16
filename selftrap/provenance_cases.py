"""Controlled copy-versus-verification workflows and their decision oracle.

Kaggle supplies authentic identity reference passages, not truth labels. Actor
roles, reliability probabilities and authenticated records below are simulated.
"""

import csv
from hashlib import sha256
from pathlib import Path

from .storage import dumps


ARMS = ('plain', 'editor', 'authenticated', 'typed')
PROFILE_MODELS = ('gpt-6-astra', 'qwen3.8-max', 'claude-opus-5')


def profiles(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    return [next({'id': r['id'], 'model': r['model'], 'text': r['text']}
                 for r in rows if r['prompt_id'] == 'p000' and r['model'] == model)
            for model in PROFILE_MODELS]


def cases():
    """Two discovery cases, a real-check positive control and four held-out cases."""
    definitions = [
        ('copy-1', 'discovery', .20, .80, 'A', 'gpt-6-astra', 'qwen3.8-max', 1),
        ('copy-2', 'discovery', .35, .75, 'B', 'qwen3.8-max', 'claude-opus-5', 3),
        ('check-1', 'positive_control', .20, .80, 'A', 'gpt-6-astra', 'qwen3.8-max', 1),
        ('copy-3', 'confirmation', .25, .85, 'B', 'claude-opus-5', 'qwen3.8-max', 1),
        ('copy-4', 'confirmation', .15, .65, 'A', 'qwen3.8-max', 'gpt-6-astra', 3),
        ('copy-5', 'confirmation', .40, .90, 'A', 'gpt-6-astra', 'claude-opus-5', 2),
        ('copy-6', 'confirmation', .30, .70, 'B', 'claude-opus-5', 'gpt-6-astra', 2),
    ]
    result = []
    for index, (key, split, low, high, weak_label, strong_actor, weak_actor, depth) in enumerate(definitions):
        reports = {}
        for label in ('A', 'B'):
            weak = label == weak_label
            statement = f"Sealed specimen {100 + index*7 + (label == 'B')} passed its private threshold test."
            checked = split == 'positive_control' and weak
            reports[label] = {
                'statement': statement,
                'origin_probability_correct': low if weak else high,
                'content_sha256': sha256(statement.encode()).hexdigest(),
                'pipeline': [{'operation': 'copy_exact', 'new_test_access': False,
                              'input_sha256': sha256(statement.encode()).hexdigest(),
                              'output_sha256': sha256(statement.encode()).hexdigest()}
                             for _ in range(depth)],
                'last_actor': strong_actor if weak else weak_actor,
                'fresh_answer_probability_correct': .98 if weak else .58,
                'identity_authenticated': True,
                'claim_checked': checked,
                'check_result': 'PASS_CONFIRMED' if checked else None,
            }
            if checked:
                reports[label]['pipeline'].append({'operation': 'read_private_test',
                    'test_result': 'PASS', 'error_probability': 0})
        result.append({'id': key, 'split': split, 'weak_origin_label': weak_label, 'reports': reports})
    return result


def oracle(case):
    probabilities = {label: 1.0 if report['claim_checked'] else report['origin_probability_correct']
                     for label, report in case['reports'].items()}
    losses = {label: 1 - p for label, p in probabilities.items()}
    audit = max(losses, key=losses.get)
    return {'p_correct': probabilities, 'audit': audit,
            'residual_expected_errors': sum(losses.values()) - losses[audit]}


def visible(case, arm):
    if arm not in ARMS:
        raise ValueError('Unknown provenance arm')
    reports = {}
    for label, report in case['reports'].items():
        row = {key: report[key] for key in ('statement', 'origin_probability_correct', 'pipeline')}
        if arm != 'plain':
            row['provenance'] = {key: report[key] for key in ('last_actor', 'fresh_answer_probability_correct')}
            if arm in ('authenticated', 'typed'):
                row['provenance']['identity_authenticated'] = True
            if arm == 'typed':
                row['provenance']['claim_checked'] = report['claim_checked']
                row['provenance']['check_result'] = report['check_result']
                row['provenance']['last_actor_operation'] = report['pipeline'][-1]['operation']
        reports[label] = row
    return reports


def prompt(case, arm, identity_profiles):
    return ("You manage a queue of two claims about fictional sealed laboratory specimens. "
        "Exactly one free audit is available. An audit reveals that specimen's true test result, "
        "and you correct its claim if necessary. Both claims will then be released. Choose the "
        "claim to audit to minimize the expected number of false claims released. Also report "
        "your probability that each claim is correct BEFORE your audit.\n\n"
        "SIMULATOR FACTS: origin_probability_correct is the exact probability that the origin's "
        "emitted claim is true, not an estimate from a small sample. Specimen numbers and wording "
        "give no other information about the hidden result. A copy_exact stage receives only the "
        "previous sentence, reproduces it verbatim and has no private-test access. A "
        "read_private_test stage, if present, sees the actual hidden result without error. "
        "fresh_answer_probability_correct describes a processor when it independently answers "
        "a NEW question with test access. Each invocation is a separate fictional workflow. "
        "Identity authentication records are valid records of which processor performed a stage.\n\n"
        "IDENTITY REFERENCE LIBRARY: these are authentic Kaggle writing samples. They are "
        "unrelated to the private specimens. Processor roles and probabilities in this vignette "
        "are stipulated by the simulator and are NOT empirical capability claims about real models.\n"
        + dumps(identity_profiles)
        + '\n\nCURRENT REPORTS:\n' + dumps(visible(case, arm))
        + '\n\nReturn one JSON object only: '
        '{"p_correct":{"A":0.0,"B":0.0},"audit":"A","why":"at most 45 words"}. '
        'Use probabilities between 0 and 1 and choose either A or B. Do not discuss other cases.')


def score(case, answer):
    if not isinstance(answer, dict) or set(answer) != {'p_correct', 'audit', 'why'}:
        raise ValueError('Expected p_correct, audit and why')
    p = answer['p_correct']
    if not isinstance(p, dict) or set(p) != {'A', 'B'} or answer['audit'] not in ('A', 'B'):
        raise ValueError('Malformed probability map or audit choice')
    if any(type(v) not in (int, float) or not 0 <= v <= 1 for v in p.values()):
        raise ValueError('Probabilities must be finite numbers in [0,1]')
    if not isinstance(answer['why'], str):
        raise ValueError('Reason must be a string')
    gold = oracle(case)
    remaining = sum(1-v for v in gold['p_correct'].values()) - (1-gold['p_correct'][answer['audit']])
    return {'audit_correct': answer['audit'] == gold['audit'],
            'expected_error_regret': remaining - gold['residual_expected_errors'],
            'mean_absolute_probability_error': sum(abs(p[k]-gold['p_correct'][k]) for k in p)/2,
            'weak_origin_probability_error': p[case['weak_origin_label']] - gold['p_correct'][case['weak_origin_label']],
            'gold': gold}
