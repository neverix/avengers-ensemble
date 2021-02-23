import pandas as pd
import catboost as cb
import xgboost as xg
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
import lightgbm as lg
from sklearn.model_selection import ParameterSampler
from scipy import stats as stats
import numpy as np
import random
from multiprocessing import Pool
import joblib
import json
from copy import deepcopy
import re
from math import inf
from functools import partial


mem = joblib.Memory("cache_dir", verbose=False)


def load_jsonl(fpath):
    with open(fpath) as jsonl:
        for line in jsonl:
            if line:
                yield json.loads(line)


def strip_muserc(jsonl):
    jsonl2 = deepcopy(jsonl)
    for line_ in jsonl2:
        if 'text' in line_['passage']:
            del line_["passage"]["text"]
        if 'did' in line_:
            del line_["did"]
        for question_ in line_["passage"]["questions"]:
            if 'question' in question_:
                del question_["question"]
            for answer_ in question_["answers"]:
                if 'text' in answer_:
                    del answer_["text"]
                if "quest" in answer_:
                    del answer_["quest"]
    return jsonl2


trans_table = dict([(ord(x), ord(y)) for x, y in zip("‘’´“”«»–-", "'''\"\"\"\"--")])


def split_sentences(text):
    if isinstance(text, list):
        return text
    return [x.strip() for x in re.split("\\(.?\\d+.?\\)", text)[1:]]


def repl_quotes(string):
    return string.strip().translate(trans_table).strip()


def it_muserc(it, prep=False, obj=False, repl=True):
    for passage in it:
        passage = passage["passage"]
        rq = repl_quotes if repl else lambda x: x
        text = rq(passage["text"].replace('\n', ''))
        if text[0] == '"' and text[-1] == '"':
            text = text[1:-1]
        if prep:
            text = split_sentences(text)

        questions = passage["questions"]
        for question in questions:
            q = rq(question["question"])
            for answer in question["answers"]:
                tup = [text, q, rq(answer["text"]), answer.get("label")]
                if obj:
                    tup.append(answer)
                yield tuple(tup)


def write_muserc(fpath, it):
    jsonl = list(load_jsonl(fpath))
    for (*_, ans), label in zip(it_muserc(jsonl, concat=False, obj=True), it):
        ans["label"] = label
    return jsonl


def write_jsonl(fpath, jsonl):
    open(fpath, 'w').write('\n'.join([json.dumps(x) for x in jsonl]))


def x_y(feats):
    if not isinstance(feats, pd.DataFrame):
        feats = pd.DataFrame(list(feats))
    return feats.drop(columns=["label"]), feats["label"]


def fit_one(model, x, y, x_test, y_test, splits, metric, params, init_params, kwargs, *args, **kw):
    sampled = list(ParameterSampler(params, 1, random_state=np.random.RandomState(random.randint(0, 100))))[0]
    mod = model(**init_params, **sampled)
    x[np.logical_not(np.isfinite(x))] = 0.
    x = x.replace([-inf, inf], np.nan).fillna(0.)
    y[np.logical_not(np.isfinite(y))] = 0
    y = y.replace([-inf, inf], np.nan).fillna(0).astype(int)
    # print(len(x))
    mod.fit(x, y, **kwargs)
    # print('done')
    probs = mod.predict_proba(x_test)
    scores = []
    for t in range(100):
        thresh = t / 100
        scores.append((run_metric(metric, probs, y_test, thresh), thresh))
    score, threshold = max(scores)
    results = {}
    for name, split in splits.items():
        results[name] = mod.predict_proba(split)
    return score, results, threshold


def fit_rs(model, init_params, x, y, x_test, y_test, splits, metric, params, n_iter=32, n_jobs=8, **kwargs):
    n_jobs = 8
    n_iter = 1
    models = [fit_one(model, x, y, x_test, y_test, splits, metric, params, init_params, kwargs)]
    # models = Pool(n_jobs).starmap(fit_one,
    #                               [(model, x, y, x_test, y_test, splits, params, init_params, kwargs)
    #                                for _ in range(n_iter)])
    score, model, threshold = max(models)
    return model, score, threshold


def f1_xgb(y_pred, dtrain):
    y_true = dtrain.get_label()
    err = 1-f1_score(y_true, (y_pred > 0).astype(np.float))
    return 'f1_err', err


@mem.cache
def xgboost_trainer(x, y, x_test, y_test, splits, metric, **kwargs):
    return fit_rs(xg.XGBClassifier, dict(verbosity=0, use_label_encoder=False, tree_method="gpu_hist"),
                  x, y, x_test, y_test, splits, metric, dict(
        n_estimators=stats.randint(50, 400),
        max_depth=stats.randint(3, 7),
        learning_rate=stats.loguniform(0.005, 0.05),
        seed=stats.randint(0, 1000)
        # early_stopping_rounds=stats.randint(50, 100)
    ), eval_set=[(x_test, y_test)], eval_metric=f1_xgb, verbose=0)


def metric_lgbm(fun, y_true, y_pred):
    dims = len(np.unique(y_true))
    if dims == 2:
        dims = 1
    preds = y_pred.reshape(dims, -1)
    return fun.__name__, run_metric(fun, preds, y_true), True
    err = fun(y_true.astype(np.int), preds)
    return fun.__name__, err, True


@mem.cache
def lgbm_trainer(x, y, x_test, y_test, splits, metric, **kwargs):
    return fit_rs(lg.LGBMClassifier, {}, x, y, x_test, y_test, splits, metric, dict(
        n_estimators=stats.randint(50, 1000),
        max_depth=stats.randint(3, 5),
        num_leaves=stats.randint(8, 84),
        early_stopping_round=stats.randint(50, 100),
        learning_rate=stats.loguniform(0.01, 1),
        random_seed=stats.randint(0, 1000)
    ), eval_set=[(x_test, y_test)], eval_metric=partial(metric_lgbm, metric), verbose=0)


def run_metric(metric, y_pred, y_true, threshold=.5):
    dims = len(np.unique(y_true))
    if len(y_pred.shape) == 1:
        if dims == 2:
            y_pred = y_pred.reshape((1, -1))
        else:
            y_pred = y_pred.reshape((dims, -1))
    if y_pred.shape[0] > y_pred.shape[1]:
        y_pred = y_pred.swapaxes(0, 1)
    if y_pred.shape[0] == 1:
        y_pred = (y_pred[0] > threshold).astype(int)
    else:
        y_pred = y_pred.argmax(axis=0).astype(int)
    return metric(y_true.astype(int), y_pred)


class CatboostMetric:
    def __init__(self, metric):
        self.metric = metric

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        arr = np.array(approxes)
        return run_metric(self.metric, arr, np.array(target).astype(int)), 1

        assert len(approxes) == 1
        assert len(target) == len(approxes[0])
        y_true = np.array(target).astype(int)
        approx = approxes[0]
        return self.metric(y_true, approx), 1

    def get_final_error(self, error, weight):
        return error


@mem.cache
def catboost_trainer(x, y, x_test, y_test, splits, metric, **kwargs):
    return fit_rs(cb.CatBoostClassifier, dict(eval_metric=CatboostMetric(metric), verbose=False), x, y, x_test, y_test, splits, metric, dict(
        iterations=stats.randint(200, 1000),
        max_depth=stats.randint(3, 6),
        learning_rate=stats.loguniform(0.01, 0.2),
        random_seed=stats.randint(0, 1000)
    ), eval_set=[(x_test, y_test)], n_jobs=8)


def train_models(x, y, x_test, y_test, splits, metric, best_feats):
    models = []
    boosts = [# xgboost_trainer, xgboost_trainer, xgboost_trainer, xgboost_trainer, xgboost_trainer,
              lgbm_trainer, lgbm_trainer, lgbm_trainer, lgbm_trainer, lgbm_trainer,
              catboost_trainer, catboost_trainer, catboost_trainer, catboost_trainer, catboost_trainer, catboost_trainer,
              # lgbm_trainer, lgbm_trainer, lgbm_trainer, lgbm_trainer, # lgbm_trainer, lgbm_trainer,
              # catboost_trainer, catboost_trainer, catboost_trainer, # catboost_trainer, catboost_trainer, catboost_trainer
              ]
    for seed, trainer in enumerate(boosts):
        print(f"  Training {trainer.__name__}...")
        model, score, threshold = trainer(x, y, x_test, y_test, splits, metric, seed=seed)
        print(f"  {trainer.__name__} - {metric.__name__}={score}, threshold={threshold}")
        results = {}
        for name, split in splits.items():
            # feats, _ = x_y(split)
            # feats = feats[best_feats]
            model_predictions = model[name]  # .predict_proba(feats)
            results[name] = model_predictions
        models.append((score, results, threshold))
    return models


@mem.cache
def ensemble_predictions(x, y, x_test, y_test, splits, metric, importances=-1., power=.25, cutoff=9):
    if importances > .0:
        pass
        '''
        print("Computing feature importances...")
        best_feats = prune_features(x, y, x_test, y_test, drop=importances)
        x = x[best_feats]
        x_test = x_test[best_feats]
        '''
    else:
        best_feats = x.columns

    for name, split in splits.items():
        splits[name] = x_y(split)[0][best_feats]

    print("Training models...")
    models = train_models(x, y,
                          x_test, y_test, splits, metric, best_feats)
    # scores = np.array([score for score, *_ in models]) - 0.9
    # weights = list(scores / scores.sum()) #
    models.sort(reverse=True, key=lambda x: x[0])
    # models = models[:1]
    # weights = [1]
    models = models[:cutoff]
    weights = []
    for _ in range(cutoff):
        weights.append(power * (1 - sum(weights)))

    # _, model, threshold = models[0] #
    # preds = [int((z[0] if len(z) == 1 else z[1]) > threshold) for z in model.predict_proba(x_y(test_feats)[0])]
    # write_jsonl("outputs/ensemble.jsonl", strip_muserc(write_muserc("datasets/MuSeRC/test.jsonl", preds)))
    # exit()

    print("Ensembling...")
    predictions_ensemble = {}
    for name, split in splits.items():
        # feats, _ = x_y(split)
        # feats = feats[best_feats]
        model_predictions = [model[name] * weight for (_, model, *_), weight in zip(models, weights)]
        predictions = np.array([np.sum(np.array(preds), axis=0) for preds in zip(*model_predictions)])
        predictions_ensemble[name] = predictions
    return predictions_ensemble


@mem.cache
def best_features(x_test, y_test):
    #
    feat = []
    for c in x_test.columns:
        xs = sorted(x_test[c])
        scores = []
        for z in xs[::len(xs)//100]:
            for n in [-1, 1]:
                y_ = (x_test[c] * n > z * n).astype(int)
                scores.append((f1_score(y_, y_test), z))
        score, thresh = max(scores)
        feat.append((score, c))
    feat.sort()
    return feat


if __name__ == '__main__':
    np.random.seed(99)
    random.seed(87)

    print("Getting features...")
    val_feats = get_features("val")
    split_thresh = int(len(val_feats) * 0.3)
    x_test, y_test = x_y(val_feats[:split_thresh])
    val_feats = val_feats[split_thresh:-split_thresh]
    x, y = x_y(val_feats)

    boost_iterations = 7
    remove_feats = 55
    best_score, best = -1., None
    all_feats = list(x.columns)
    test_feats = get_features("test")
    for i in range(boost_iterations):
        print(f"Boost iteration {i+1}/{boost_iterations}")
        sel = random.sample(all_feats, len(all_feats) - remove_feats)
        print("Features:", sel)
        resel = lambda x: [{s: z[s] for s in sel + ["label"]} for z in x]

        print(" Building model...")
        splits = {fn: resel(get_features(fn)) for fn in ["test", "val"]}
        predictions_ensemble = ensemble_predictions(x[sel], y, x_test[sel], y_test, splits, power=.4, cutoff=11)

        print(" Evaluating...")
        noise = 0.
        predictions_ensemble["val"] = [z + random.gauss(0, noise) for z in predictions_ensemble["val"]]
        scores = []
        for t in range(0, 100):
            thresh = t / 100
            y_ = [int(z > thresh) for z in predictions_ensemble["val"][:split_thresh]]
            scores.append((f1_score(y_, y_test), thresh))
        score, threshold = max(scores)
        print(" Combined F1:", score)
        print(" Threshold:", threshold)

        # threshold = 0.5
        if score > best_score:
            best_score = score
            best = predictions_ensemble, threshold
    predictions_ensemble, threshold = best
    print("Best F1", best_score)
    preds = [int(z > threshold) for z in predictions_ensemble["test"]]
    print("Positives", sum(preds[:len(preds) // 2]))
    print("Negatives", len(preds) // 2 - sum(preds[:len(preds) // 2]))
    print("Length", len(preds) // 2)
    write_jsonl("outputs/ensemble.jsonl", strip_muserc(write_muserc("datasets/MuSeRC/test.jsonl", preds)))

    print("Done!")