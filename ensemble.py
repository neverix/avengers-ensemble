import sys

import data
import pandas as pd
from autogluon.tabular import TabularPredictor, TabularDataset
import utils
import shutil
import ast
import itertools
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import os
from tqdm import tqdm
from cache import mem
import warnings
from muserc import write_muserc, strip_muserc
import resemble
from sklearn.model_selection import train_test_split
import numpy as np
import copy
import random
from datetime import datetime


@mem.cache
def subset_sum(dct, target):
    for r in range(len(dct) + 1, 0, -1):
        for comb in itertools.combinations(dct, r):
            if sum(dct[w] for w in comb) == target:
                return comb


def read_split(datasets, model, split):
    scores_file = f"scores/{model}.{split}.scores"
    if split == "val":
        try:
            open(scores_file)
        except FileNotFoundError:
            split = "dev"
            scores_file = f"scores/{model}.{split}.scores"
    lines = list(open(scores_file))
    result = []
    for line in lines:
        line = line.strip()
        try:
            result.append(ast.literal_eval(line))
        except (SyntaxError, ValueError) as e:
            result.append(tuple(map(float, line.split())))
    lines = result
    lines = [x if isinstance(x, tuple) else (x,) for x in lines]

    length = len(lines)
    lens = {key: len(val) for key, val in datasets.items()}
    # print("Solving subset sum...")
    for length_ in range(length+1, length-2, -1):
        comb = subset_sum(lens, length_)
        if comb is not None:
            # print(model, length, length_)
            break
    assert comb is not None
    # print("Solved!")
    funs = [fn for fn in data.data_funs if fn in comb]
    # hack
    if model.startswith("xlm/anli"):
        # The models trained on all datasets do worse on each of them,
        # but the sample-limited RWSD benefits from the extra data
        funs = [data.read_rwsd]

    results = {}
    for fn in funs:
        datas = datasets[fn]
        keys = sorted(datas.keys())
        results[fn] = dict(zip(keys, lines[:len(datas)]))
        lines = lines[len(datas):]
    return results


def process_terra(_dataset, preds, _probs):
    return [{"idx": k, "label": ["not_entailment", "entailment"][int(v)]} for k, v in preds.items()]


def process_lidirus(_dataset, preds, _probs):
    return [{"idx": k, "label": ["not_entailment", "entailment"][int(v)]} for k, v in preds.items()]


def process_rcb(_dataset, preds, _probs):
    return [{"idx": k, "label": ["contradiction", "neutral", "entailment"][int(v)]} for k, v in preds.items()]


def process_russe(_dataset, preds, _probs):
    return [{"idx": k, "label": [False, True][int(v)]} for k, v in preds.items()]


def process_parus(_dataset, _preds, probs):
    vals = list(probs.values())
    result = []
    for i in range(len(vals) // 2):
        result.append({"idx": i, "label": int(vals[i*2+0] > vals[i*2+1])})
    return result


def process_danetqa(dataset, preds, probs):
    return [{"idx": k, "label": bool(v)} for k, v in preds.items()]
    _, source = dataset
    _, dataset, _ = source[(data.read_danetqa, "test")]
    questions = {}
    for sample, prob in zip(dataset.values(), probs.values()):
        question = sample["question"]
        if question not in questions:
            questions[question] = []
        questions[question].append(prob)
    questions = {k: sum(v) / len(v) for k, v in questions.items()}
    # probs = {k: (questions[s["question"]]) for (k, v), s in zip(probs.items(), dataset.values())}
    return [{"idx": k, "label": bool(v > 0.5)} for k, v in probs.items()]


def process_muserc(_dataset, preds, _probs):
    return strip_muserc(write_muserc("datasets/MuSeRC/test.jsonl", [int(x) for x in preds.values()]))
    # splits, source = dataset
    # src, datas, data_ = source[(data.read_muserc, "test")]
    # print(src.keys(), src[5])
    # exit()


def process_rwsd(_dataset, preds, _probs):
    return [{"idx": int(k), "label": [False, True][int(v)]} for k, v in preds.items()]


models = ["xlm/anli", "xlm/anli-terra", "xlm/anli-all", "xlm/anli-all-x", "xlm/anli-rcb",
          "zero-norm/super",
          "zero-norm/super-rcb", "silver/silver", "golden/1", "golden/2", "golden/3", "golden/4",
          "zero-norm/super-qa", "golden/danetqa", "golden/danetqa-better", "zero-norm/super-proc", "qa/en-azero",
          "golden/danetqa-5000", "platinum/1", "zero56/zero", "platinum/1-fp", "platinum/1r", "platinum/1rs",
          "gpt/medium-mrc", "golden/mix", "word/word", "qa/xlm", "qa/zero", "zero-norm/super-plus", "qa/en-zero",
          "qa/en-altzero", "zerode/del", "train/xlm", "train/bert", "zerode/delx",  # "zerode/xlarge",
          "zerode/xlm", "golden/mix-5000", "56/feats", "zerode/en", "zerode/de", "zerode/dex",  # "golden/nop",
          "golden-nop/nop", "zerode/dexx", "zerode/den", "golden-nop/mix", "golden-yep/mix", "golden-yep/nop",
          "zero/zero", "zero-alt/zero", "zero-alt/zero83", "zero-norm/zero",
          # "mbert/mbert",
          "train/xlm-multirc", "train/xlm-multirc-better", "qa/en-albzero", "train/xlm-danetqa", "train/xlm-both",
          'train/xlm-many', "train/rb-both", "train/rb-last", "train/al-both", "train/de", "train/rb-long",
          "train/dex", "train/alb-both", "train/mt5", "train/mt5-long", "train/mt5-long-2",
          # "process/raw-small", "process/raw-base", "process/raw-large", "process/raw-3B",
          "process/rawe-small", "process/rawe-base", "process/rawe-large", "process/rawe-3B",
          "process/none-small", "process/none-base", "process/none-large", "process/none-3B",
          "process/all-small", "process/all-base", "process/all-large", "process/all-3B",
          "train/large", "11/11b", "train/large-nli", "11/11-train", "11/11-long",
          "process/noq-small", "process/noq-base", "process/noq-large", "process/noq-3B",
          "process/bcq-small", "process/bcq-base", "process/bcq-large", "process/bcq-3B",
          "11/11-longer", "11/11-longest", "11/11-longerest", "11/11a-f", "11/11-f",
          "11/11a-long", "11/11a-longer",
          # "new-rob-large/roblexp3_rte",  # "new-rob-large/roblexp2_rte",  # "new-rob-large/roblexp_rte",
          # "new-rob-large/roblexp6_rte", "new-rob-large/roblexp5_rte", "new-rob-large/roblexp4_rte",
          # "new-rob-large/roblexp9_rte", "new-rob-large/roblexp8_rte", "new-rob-large/roblexp7_rte",  #
          "new-rob-large/xwl_wic", "new-rob-large/xwl_wsd"
          ]
for step in ["1001200", "1003000", "1004800", "1006000", "1007800", "1010800", "1013200", "1016800", "1019200"][-1:]:
    models.append(f"all/all-{step}")
files = set(x.split('.')[0] for x in os.listdir("scores/all-step"))
for file in files:
    models.append(f"all-step/{file}")
datasets = {
    data.read_muserc: (process_muserc, "MuSeRC", "f1"),
    data.read_danetqa: (process_danetqa, "DaNetQA", "acc"),
    data.read_rwsd: (process_rwsd, "RWSD", "acc"),
    data.read_rcb: (process_rcb, "RCB", "acc"),
    data.read_terra: (process_terra, "TERRa", "acc"),
    data.read_lidirus: (process_lidirus, "LiDiRus", "mcc"),
    data.read_russe: (process_russe, "RUSSE", "acc"),
    data.read_parus: (process_parus, "PARus", "acc"),
}
metrics = dict(
    f1=f1_score,
    acc=accuracy_score,
    mcc=matthews_corrcoef,
)
boost_iterations = 1
keep_feats = 1.
corr_thresh = 1.


def make_feats(dataset):
    preds = {}
    for split in ("val", "test"):
        datasets = {fn: fn(split) for fn in data.data_funs}
        for model in models:
            try:
                preds[(model, split)] = read_split(datasets, model, split)
            except FileNotFoundError as e:
                print(e, file=sys.stderr)
    print("Got features.")

    splits, source = dataset
    total = {}
    for fn in data.data_funs:
        splits = {}
        for split in ("val", "test"):
            data1, data2, data3 = source[(fn.__name__, split)]
            results = {key: {} for key in data2}
            for model in models:
                if (model, split) not in preds:
                    continue
                result = preds[(model, split)]
                if fn not in result:
                    continue
                result = result[fn]
                for key in result:
                    for i, e in enumerate(result[key]):
                        feat_name = f"{model}_{i}"
                        results[key][feat_name] = e
                    results[key]["label"] = data2[key].get("label", 0)
            splits[split] = results
        total[fn] = splits
    return total


def x_y(feats):
    feats = pd.DataFrame(feats.values())
    return feats.drop(columns=["label"]), feats["label"]


def ensemble_predictions(train, splits, metric, feats=None):
    x, y = x_y(train)
    if feats is None:
        feats = list(x.columns)

    x = x[feats]
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=56, shuffle=False, test_size=0.3)

    og_splits = copy.deepcopy(splits)
    for split in splits:
        if "label" not in feats:
            feats.append("label")
        x, y = x_y(splits[split])
        x["label"] = y
        splits[split] = x[feats]
    # splits = {key: x_y(split) for key, split in splits.items()}
    cols = list(set(x.columns) - {"label"})

    best_score, best = -1, None
    bests = []
    for i in range(boost_iterations):
        print("Boosting,", i+1, "out of", boost_iterations)
        sel = random.sample(cols, int(len(cols) * keep_feats))
        # resel = lambda m: {k: {s: z[s] for s in sel + ["label"]} for k, z in m.items()}
        new_splits = {fn: split[sel +
                                (["label"] if "label" in split.columns else [])].copy()
                      for fn, split in splits.items()}
        predictions_ensemble, score = resemble.ensemble_predictions(x_train[sel].copy(), y_train,
                                                                    x_test[sel].copy(), y_test,
                                                                    new_splits, metrics[metric])
        bests.append(score)
        if score > best_score:
            best_score = score
            best = predictions_ensemble
        print("Boosting result:", score)
    print("Best score:", best_score, "out of", *bests)
    predictions_ensemble = best

    for name, probs in predictions_ensemble.items():
        split = og_splits[name]
        classes, probs = np.argmax(probs, axis=1), probs
        if probs.shape[1] == 2:
            probs = probs[:, :1]
        predictions_ensemble[name] = dict(zip(split.keys(), [int(x) for x in classes])), dict(zip(split.keys(), [tuple(map(float, x)) for x in probs]))
    return predictions_ensemble


def ensemble_predictions_ag(train, splits, metric, feats=None):
    x, y = x_y(train)
    if feats is None:
        feats = list(x.columns)

    print("Training...")
    x = x[feats]
    x["label"] = y
    train_data = TabularDataset(x)  # task.Dataset(x)
    eval_metric = dict(
        f1="F1",
        mcc="MCC",
        acc="Accuracy"
    )[metric]
    predictor = TabularPredictor(label="label", eval_metric=metric) \
        .fit(presets="best_quality", train_data=train_data, hyperparameters={'GBM': {}, 'CAT': dict(
        eval_metric=eval_metric
    ), 'RF': {}})
    print("Computing...")
    predictions_ensemble = {}
    for name, split in splits.items():
        feats, y_true = x_y(split)
        # feats = feats[best_feats]
        # model_predictions = [model[name] * weight for (_, model, *_), weight in zip(models, weights)]
        # predictions = [float(sum([x[1] if len(x) > 1 else x[0] for x in preds])) for preds in zip(*model_predictions)]
        y_pred = predictor.predict(feats)
        y_prob = [float(x) if (isinstance(x, int) or isinstance(x, float) or x.size < 2)
                  else [float(y) for y in x]
                  for x in predictor.predict_proba(feats)]
        if split == "val":
            print(predictor.evaluate_predictions(y_true=y_true, y_pred=y_pred, auxiliary_metrics=True))
        predictions_ensemble[name] = dict(zip(split.keys(), y_pred)), dict(zip(split.keys(), y_prob))
    return predictions_ensemble


# ensemble_predictions = ensemble_predictions_ag


def build_model(dataset, feats, fn):
    fun, _, metric, *_ = datasets[fn]

    x, y = x_y(feats[fn]["val"])
    corr = x.corr().abs()
    cols = ['_'.join(x.split('_')[:-1]) for x in corr.columns.tolist()]
    top = []
    for x, a in enumerate(cols):
        for y, b in enumerate(cols[:x]):
            if a == b:
                continue
            c = corr.iloc[x, y]
            if c != 1:
                top.append((c, a, b))
    top.sort()
    allowed = corr.columns.tolist()
    while True and top:
        score, a, b = top.pop()
        if score < corr_thresh:
            break
        remove = a
        top = [x for x in top if remove not in (x[1], x[2])]
        allowed = [col for col in allowed if '_'.join(col.split('_')[:-1]) != remove]

    preds = ensemble_predictions(feats[fn]["val"], feats[fn], metric=metric, feats=allowed)
    return fun(dataset, *preds["test"])


def best_features(x_test, y_test, metric=accuracy_score):
    y_test = y_test.fillna(0)
    all_feats = x_test.columns
    real_feats = set('_'.join(feat.split('_')[:-1]) if feat[-1].isdigit() and '_' in feat else feat for feat in all_feats)
    feats = []
    for feat in tqdm(real_feats):
        related = [x for x in all_feats if x.startswith(feat + '_')]
        x_rel = x_test[related].fillna(0)
        model = LogisticRegression(C=1e56)  # RandomForestClassifier(n_estimators=16, max_depth=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_rel, y_test)
        probs = model.predict(x_rel)
        feats.append((metric(probs, y_test), feat))
    feats.sort()
    return feats


if __name__ == '__main__':
    np.random.seed(56)
    random.seed(56)

    print("Loading data...")
    dataset = data.load_all(verbose=True)
    print("Making features...")
    feats = make_feats(dataset)
    print("Features done, finding best...")

    for fn in datasets:
        print(datasets[fn][1])
        val_df = feats[fn]["val"]
        x, y = x_y(val_df)
        print("Getting best features...")
        feat = best_features(x, y, metric=metrics[datasets[fn][-1]])
        for score, c in feat:
            print(f" {c}: {score}")
    # exit()

    for fn, (processor, name, *_) in datasets.items():
        print(f"Training {name}...")
        preds = build_model(dataset, feats, fn)
        print(f"Writing {name}...")
        utils.write_jsonl(f"outputs/{name}.jsonl", preds)
    print("Archiving...")
    shutil.make_archive(f"out/outputs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}", "zip", "outputs/")
    print("Done!")
