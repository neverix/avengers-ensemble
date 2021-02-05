import data
import pandas as pd
from autogluon.tabular import TabularPrediction as task
import utils
import shutil
import ast
import itertools


def subset_sum(dct, target):
    for r in range(1, len(dct) + 1):
        for comb in itertools.combinations(dct, r):
            if sum(dct[w] for w in comb) == target:
                return comb


def read_split(model, split):
    lines = list(map(ast.literal_eval, open(f"scores/{model}.{split}.scores")))

    length = len(lines)
    datasets = {fn: fn(split) for fn in data.data_funs}
    lens = {key: len(val) for key, val in datasets.items()}
    print("Solving subset sum...")
    for length in range(length+1, length-2, -1):
        comb = subset_sum(lens, length)
        if comb is not None:
            break
    assert comb is not None
    print("Solved!")
    funs = [fn for fn in data.data_funs if fn in comb]

    results = {}
    for fn in funs:
        dataset = fn(split)
        datas = data.preprocess_dataset(dataset)
        keys = sorted(datas.keys())
        results[fn] = dict(zip(keys, lines[:len(datas)]))
        lines = lines[len(datas):]
    return results


def process_terra(_dataset, preds):
    return [{"idx": k, "label": ["not_entailment", "entailment"][int(v)]} for k, v in preds.items()]


def process_lidirus(_dataset, preds):
    return [{"idx": k, "label": ["not_entailment", "entailment"][int(v)]} for k, v in preds.items()]


def process_rcb(_dataset, preds):
    return [{"idx": k, "label": ["contradiction", "neutral", "entailment"][int(v)]} for k, v in preds.items()]


models = ("xlm/anli", "xlm/anli-terra", "xlm/anli-all", "xlm/anli-all-x", "xlm/anli-rcb", "zero/zero", "mbert/mbert")[:-1]
datasets = {
    data.read_terra: (process_terra, "TERRa", "acc"),
    data.read_rcb: (process_rcb, "RCB", "acc"),
    data.read_lidirus: (process_lidirus, "LiDiRus", "mcc"),
}


def make_feats(dataset):
    preds = {}
    for split in ("val", "test"):
        for model in models:
            preds[(model, split)] = read_split(model, split)

    splits, source = dataset
    total = {}
    for fn in data.data_funs:
        splits = {}
        for split in ("val", "test"):
            data1, data2, data3 = source[(fn, split)]
            results = {key: {} for key in data2}
            for model in models:
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


def ensemble_predictions(train, splits, metric):
    print("Training...")
    x, y = x_y(train)
    x["label"] = y
    train_data = task.Dataset(x)
    predictor = task.fit(presets="best_quality", train_data=train_data, eval_metric=metric, label="label", hyperparameters={'GBM': {}, 'CAT': dict(
        eval_metric=dict(
            f1="F1",
            mcc="MCC",
            acc="Accuracy"
        )[metric]
    ), 'RF': {}})
    print("Computing...")
    predictions_ensemble = {}
    for name, split in splits.items():
        feats, y_true = x_y(split)
        # feats = feats[best_feats]
        # model_predictions = [model[name] * weight for (_, model, *_), weight in zip(models, weights)]
        # predictions = [float(sum([x[1] if len(x) > 1 else x[0] for x in preds])) for preds in zip(*model_predictions)]
        y_pred = predictor.predict(feats)
        if split == "val":
            print(predictor.evaluate_predictions(y_true=y_true, y_pred=y_pred, auxiliary_metrics=True))
        predictions_ensemble[name] = dict(zip(split.keys(), y_pred))
    return predictions_ensemble


def build_model(dataset, feats, fn):
    fun, _, metric, *_ = datasets[fn]
    preds = ensemble_predictions(feats[fn]["val"], feats[fn], metric=metric)
    return fun(dataset, preds["test"])


if __name__ == '__main__':
    dataset = data.load_all(verbose=True)
    feats = make_feats(dataset)
    for fn, (processor, name, *_) in datasets.items():
        preds = build_model(dataset, feats, fn)
        print(f"Writing {name}...")
        utils.write_jsonl(f"outputs/{name}.jsonl", preds)
    shutil.make_archive("outputs", "zip", "outputs/")
