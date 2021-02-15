import data
import pandas as pd
from autogluon.tabular import TabularPrediction as task
import utils
import shutil
import ast
import itertools
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from cache import mem
import warnings
from muserc import write_muserc, strip_muserc


@mem.cache
def subset_sum(dct, target):
    for r in range(len(dct) + 1, 0, -1):
        for comb in itertools.combinations(dct, r):
            if sum(dct[w] for w in comb) == target:
                return comb


def read_split(model, split):
    lines = list(map(ast.literal_eval, open(f"scores/{model}.{split}.scores")))
    lines = [x if isinstance(x, tuple) else (x,) for x in lines]

    length = len(lines)
    datasets = {fn: fn(split) for fn in data.data_funs}
    lens = {key: len(val) for key, val in datasets.items()}
    print("Solving subset sum...")
    for length_ in range(length+1, length-1, -1):
        comb = subset_sum(lens, length_)
        if comb is not None:
            # print(model, length, length_)
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
        result.append({"idx": i, "label": int(vals[i*2+1] > vals[i*2+0])})
    return result


def process_danetqa(_dataset, preds, _probs):
    return [{"idx": k, "label": [False, True][int(v)]} for k, v in preds.items()]


def process_muserc(_dataset, preds, _probs):
    return strip_muserc(write_muserc("datasets/MuSeRC/test.jsonl", preds))
    # splits, source = dataset
    # src, datas, data_ = source[(data.read_muserc, "test")]
    # print(src.keys(), src[5])
    # exit()


def process_rwsd(_dataset, preds, _probs):
    return [{"idx": k, "label": [False, True][int(v)]} for k, v in preds.items()]


models = ("xlm/anli", "xlm/anli-terra", "xlm/anli-all", "xlm/anli-all-x", "xlm/anli-rcb", "zero-norm/super",
          "zero-norm/super-rcb", "silver/silver", "golden/1", "golden/2", "golden/3", "golden/4",
          "zero-norm/super-qa", "golden/danetqa", "golden/danetqa-better", "zero-norm/super-proc", "qa/en-azero",
          "golden/danetqa-5000", "platinum/1", "zero56/zero", "platinum/1-fp", "platinum/1r", "platinum/1rs",
          "gpt/medium-mrc", "golden/mix", "word/word", "qa/xlm", "qa/zero", "zero-norm/super-plus", "qa/en-zero",
          "qa/en-altzero", "zerode/xlarge",
          "zero/zero", "zero-alt/zero", "zero-alt/zero83", "zero-norm/zero", "mbert/mbert")[:-1]
datasets = {
    # data.read_rwsd: (process_rwsd, "RWSD", "acc"),
    data.read_muserc: (process_muserc, "MuSeRC", "f1"),
    # data.read_terra: (process_terra, "TERRa", "acc"),
    # data.read_rcb: (process_rcb, "RCB", "acc"),
    # data.read_lidirus: (process_lidirus, "LiDiRus", "mcc"),
    # data.read_russe: (process_russe, "RUSSE", "acc"),
    # data.read_parus: (process_parus, "PARus", "acc"),
    data.read_danetqa: (process_danetqa, "DaNetQA", "acc"),
}
metrics = dict(
    f1=f1_score,
    acc=accuracy_score,
    mcc=matthews_corrcoef,
)


def make_feats(dataset):
    preds = {}
    for split in ("val", "test"):
        for model in models:
            preds[(model, split)] = read_split(model, split)
    print("Got features.")

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
        y_prob = [float(x) if x.size < 2 else [float(y) for y in x] for x in predictor.predict_proba(feats)]
        if split == "val":
            print(predictor.evaluate_predictions(y_true=y_true, y_pred=y_pred, auxiliary_metrics=True))
        predictions_ensemble[name] = dict(zip(split.keys(), y_pred)), dict(zip(split.keys(), y_prob))
    return predictions_ensemble


def build_model(dataset, feats, fn):
    fun, _, metric, *_ = datasets[fn]
    preds = ensemble_predictions(feats[fn]["val"], feats[fn], metric=metric)
    return fun(dataset, *preds["test"])


def best_features(x_test, y_test, metric=accuracy_score):
    y_test = y_test.fillna(0)
    all_feats = x_test.columns
    real_feats = set('_'.join(feat.split('_')[:-1]) if feat[-1].isdigit() and '_' in feat else feat for feat in all_feats)
    feats = []
    for feat in real_feats:
        related = [x for x in all_feats if x.startswith(feat + '_')]
        x_rel = x_test[related].fillna(0)
        model = LogisticRegression()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_rel, y_test)
        probs = model.predict(x_rel)
        feats.append((metric(probs, y_test), feat))
    feats.sort()
    return feats


if __name__ == '__main__':
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
        preds = build_model(dataset, feats, fn)
        print(f"Writing {name}...")
        utils.write_jsonl(f"outputs/{name}.jsonl", preds)
    print("Archiving...")
    shutil.make_archive("outputs", "zip", "outputs/")
    print("Done!")
