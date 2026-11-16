"""CPU baseline and prompt-disjoint audit for Kaggle Name That LLM.

No leaderboard submission and no paid model calls. Model labels identify the
original generator; they do not label truth, honesty or answer quality.
"""

import argparse
import csv
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import re
from time import perf_counter

import numpy as np

from .storage import digest, dumps, read_json, utc_now, write_json


SEED = 20261116


def download_data(directory):
    """Read-only competition download; never join, accept terms or submit."""
    import zipfile
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files("name-that-llm", path=str(directory), quiet=False, unzip=False)
        archive = directory / "name-that-llm.zip"
        with zipfile.ZipFile(archive) as files:
            if set(files.namelist()) != {"train.csv", "test.csv", "sample_submission.parquet"}:
                raise ValueError("Unexpected archive members")
            for name in files.namelist():
                (directory / name).write_bytes(files.read(name))
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", getattr(exc, "status", None))
        raise RuntimeError(f"Kaggle download failed: {type(exc).__name__}, status={status}; credentials not logged") from None
    write_json(directory.parent / "download-manifest.json", {"at": utc_now(), "competition": "name-that-llm",
        "source": "https://www.kaggle.com/competitions/name-that-llm/data",
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p.read_bytes()).hexdigest()}
                  for p in directory.iterdir() if p.is_file()}})


def load_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"id", "prompt_id", "text", "model"}
    if not rows or not required <= rows[0].keys():
        raise ValueError("Need labelled train.csv with id,prompt_id,text,model")
    if any(not all(r.get(k) for k in required) for r in rows):
        raise ValueError("Empty required field")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate passage IDs")
    return rows


def split_groups(rows, seed=SEED):
    groups = sorted({r["prompt_id"] for r in rows})
    if len(groups) < 10:
        raise ValueError("Need at least ten prompt groups")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    size = len(groups) // 5
    chosen = {"test": set(groups[:size]), "calibration": set(groups[size:2*size]),
              "train": set(groups[2*size:])}
    return {name: [r for r in rows if r["prompt_id"] in ids] for name,ids in chosen.items()}


def normalize_typography(text):
    """A modest surface perturbation, not a semantic-equivalence certificate."""
    table = str.maketrans({"—": " - ", "–": "-", "“": '"', "”": '"', "‘": "'", "’": "'",
                          "\u00a0": " ", "\u202f": " "})
    return re.sub(r"\s+", " ", text.translate(table)).strip()


def surface_features(text):
    words = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
    n = max(1, len(text)); w = max(1, len(words))
    lengths = [len(t) for t in words] or [0]
    features = {"words": len(words), "characters": len(text),
                "mean_word_length": float(np.mean(lengths)),
                "sentence_marks_per_word": sum(text.count(c) for c in ".!?") / w,
                "uppercase_fraction": sum(c.isupper() for c in text) / n,
                "digit_fraction": sum(c.isdigit() for c in text) / n,
                "newline_fraction": text.count("\n") / n}
    for character in (",", ";", ":", "—", "-", "(", ")", '"', "“", "*", "!", "?"):
        features[f"punctuation_{ord(character)}"] = text.count(character) / n
    return features


def temperature(logits, labels, classes):
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp
    positions = np.array([np.flatnonzero(classes == y)[0] for y in labels])
    def objective(log_t):
        scores = logits / np.exp(log_t)
        return float(np.mean(logsumexp(scores, axis=1) - scores[np.arange(len(labels)), positions]))
    result = minimize_scalar(objective, bounds=(-2.3, 2.3), method="bounded")
    return float(np.exp(result.x))


def metrics(probabilities, labels, classes):
    from sklearn.metrics import balanced_accuracy_score, accuracy_score
    positions = np.array([np.flatnonzero(classes == y)[0] for y in labels])
    prediction = probabilities.argmax(axis=1)
    correct = prediction == positions
    confidence = probabilities.max(axis=1)
    ece = 0.0
    for low, high in zip(np.linspace(0,1,11)[:-1], np.linspace(0,1,11)[1:]):
        selected = (confidence > low) & (confidence <= high)
        if selected.any():
            ece += selected.mean() * abs(confidence[selected].mean() - correct[selected].mean())
    truth = np.eye(len(classes))[positions]
    return {"accuracy": float(accuracy_score(labels, classes[prediction])),
            "balanced_accuracy": float(balanced_accuracy_score(labels, classes[prediction])),
            "nll": float(-np.log(np.maximum(probabilities[np.arange(len(labels)),positions],1e-12)).mean()),
            "brier_sum_over_classes": float(((probabilities-truth)**2).sum(axis=1).mean()),
            "ece_10_bins": float(ece), "mean_top1_confidence": float(confidence.mean())}


def grouped_interval(probabilities, rows, classes, repeats=1000):
    groups = sorted({r["prompt_id"] for r in rows})
    position = {g:i for i,g in enumerate(groups)}
    correct = np.zeros((len(groups),len(classes)))
    total = correct.copy()
    class_index = {c:i for i,c in enumerate(classes)}
    for row, pred in zip(rows, probabilities.argmax(axis=1)):
        g,c = position[row["prompt_id"]],class_index[row["model"]]
        total[g,c] += 1
        correct[g,c] += pred == c
    rng = np.random.default_rng(SEED+1)
    values = []
    for _ in range(repeats):
        sampled = rng.integers(0,len(groups),len(groups))
        denom = total[sampled].sum(axis=0)
        values.append(float((correct[sampled].sum(axis=0)/denom).mean()))
    return [float(x) for x in np.quantile(values,[.025,.975])]


def run(data, output):
    from sklearn import __version__ as sklearn_version
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import StandardScaler
    from scipy.special import softmax
    from threadpoolctl import threadpool_limits
    from tqdm import tqdm
    import joblib

    data, output = Path(data), Path(output)
    rows = load_rows(data)
    split = split_groups(rows)
    spec = {"protocol": "model-style-prompt-disjoint-v1", "seed": SEED,
            "data_sha256": sha256(data.read_bytes()).hexdigest(),
            "source_hash": digest(Path(__file__).read_text()), "sklearn": sklearn_version,
            "splits": {k: {"prompt_ids": sorted({r['prompt_id'] for r in v}), "ids": [r['id'] for r in v]} for k,v in split.items()},
            "models": ["char_tfidf_ovr_logistic", "surface_features_logistic"],
            "char_parameters": {"ngrams": [3,5], "max_features": 50000, "min_df": 3, "C": 3},
            "calibration": "One scalar softmax temperature fitted ONLY on separate calibration prompts.",
            "interval": "1000 prompt-cluster bootstrap draws; these are resamples, not new model trials.",
            "scope": "Source attribution only. No truth, intent, honesty or epistemic-reliability labels exist here. Surface features measure lengths and punctuation; char ngrams also capture lexical/content cues."}
    manifest = output / "design.json"
    if manifest.exists():
        if read_json(manifest)["spec_hash"] != digest(spec):
            raise ValueError("Analysis or split changed; choose a new output directory")
        if (output/"summary.json").exists():
            print(dumps(read_json(output/"summary.json")))
            return
    else:
        write_json(manifest,{"at":utc_now(),"before_fitting":True,"spec_hash":digest(spec),"spec":spec})
    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(len(re.findall(r"\b\w+\b",row["text"])))
    profile = {"rows":len(rows),"prompt_groups":len({r['prompt_id'] for r in rows}),
               "models":dict(Counter(r['model'] for r in rows)),
               "group_sizes":dict(Counter(Counter(r['prompt_id'] for r in rows).values())),
               "duplicate_texts":len(rows)-len({r['text'] for r in rows}),
               "word_counts":{k:{"mean":float(np.mean(v)),"median":float(np.median(v)),"min":min(v),"max":max(v)} for k,v in by_model.items()}}
    write_json(output/"profile.json",profile)
    train, calibration, test = (split[s] for s in ("train","calibration","test"))
    summary = {"counts":{k:len(v) for k,v in split.items()},"chance_balanced_accuracy":1/len(by_model),"results":{}}
    with threadpool_limits(limits=2):
        for kind in tqdm(spec["models"],desc="Fit two small CPU attribution baselines"):
            started = perf_counter()
            if kind.startswith("char"):
                vector = TfidfVectorizer(analyzer="char",ngram_range=(3,5),min_df=3,max_features=50000,sublinear_tf=True,dtype=np.float32)
                raw_train,raw_cal,raw_test = ([r['text'] for r in part] for part in (train,calibration,test))
            else:
                vector = DictVectorizer(sparse=True)
                raw_train,raw_cal,raw_test = ([surface_features(r['text']) for r in part] for part in (train,calibration,test))
            x_train = vector.fit_transform(raw_train)
            x_cal,x_test = vector.transform(raw_cal),vector.transform(raw_test)
            scale = StandardScaler(with_mean=False) if not kind.startswith("char") else None
            if scale:
                x_train=scale.fit_transform(x_train);x_cal=scale.transform(x_cal);x_test=scale.transform(x_test)
            classifier = OneVsRestClassifier(LogisticRegression(C=3,solver="liblinear",max_iter=1000,random_state=SEED))
            classifier.fit(x_train,[r['model'] for r in train])
            classes=classifier.classes_
            t=temperature(classifier.decision_function(x_cal),[r['model'] for r in calibration],classes)
            probabilities=softmax(classifier.decision_function(x_test)/t,axis=1)
            result={**metrics(probabilities,[r['model'] for r in test],classes),
                    "balanced_accuracy_prompt_bootstrap_95":grouped_interval(probabilities,test,classes),"temperature":t,
                    "seconds":perf_counter()-started,"features":x_train.shape[1]}
            normalized=[normalize_typography(r['text']) for r in test]
            x_normal=vector.transform(normalized if kind.startswith("char") else [surface_features(s) for s in normalized])
            if scale:x_normal=scale.transform(x_normal)
            changed=softmax(classifier.decision_function(x_normal)/t,axis=1)
            result["normalized_typography"]={**metrics(changed,[r['model'] for r in test],classes),
                "fraction_top1_changed":float((changed.argmax(axis=1)!=probabilities.argmax(axis=1)).mean())}
            summary['results'][kind]=result
            model_path=output/f"{kind}.joblib"
            joblib.dump({'vectorizer':vector,'scaler':scale,'classifier':classifier,'temperature':t,'classes':classes},model_path)
            with (output/f"{kind}-test-predictions.jsonl").open('w') as stream:
                for row,p in zip(test,probabilities):
                    stream.write(dumps({'id':row['id'],'prompt_id':row['prompt_id'],'true_model':row['model'],
                        'probabilities':{c:float(v) for c,v in zip(classes,p)}})+'\n')
    write_json(output/"summary.json",summary)
    print(dumps(summary))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('runs/kaggle/name-that-llm/raw/train.csv'))
    parser.add_argument('--output',type=Path,default=Path('runs/kaggle/name-that-llm/baseline-1'))
    parser.add_argument('--download',action='store_true',help='Use the configured Kaggle token for a read-only download first')
    args=parser.parse_args()
    if args.download:
        download_data(args.data.parent)
    run(args.data,args.output)
