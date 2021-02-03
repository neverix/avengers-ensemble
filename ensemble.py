import data
import pandas as pd
from autogluon.tabular import TabularPrediction as task
import utils
import shutil


def process_terra(_dataset, preds):
    return [{"idx": k, "label": int(v)} for k, v in preds.items()]


models = ("mbert/mbert",)
processors = {
    data.read_terra: process_terra
}
names = {
    data.read_terra: "TERRa"
}


def make_feats(dataset):
    preds = {}
    for split in ("val", "test"):
        for model in models:
            preds[(model, split)] = data.read_split(model, split)

    splits, source = dataset
    total = {}
    for fn in data.data_funs:
        splits = {}
        for split in ("val", "test"):
            data1, data2, data3 = source[(fn, split)]
            results = {key: {} for key in data2}
            for model in models:
                result = preds[(model, split)][fn]
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


def ensemble_predictions(train, splits):
    print("Training...")
    x, y = x_y(train)
    x["label"] = y
    train_data = task.Dataset(x)
    predictor = task.fit(presets="best_quality", train_data=train_data, eval_metric="f1", label="label", hyperparameters={'GBM': {}, 'CAT': {}, 'RF': {}})
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
    preds = ensemble_predictions(feats[fn]["val"], feats[fn])
    return processors[fn](dataset, preds["test"])


if __name__ == '__main__':
    dataset = data.load_all(verbose=True)
    feats = make_feats(dataset)
    for fn in processors.keys():
        preds = build_model(dataset, feats, data.read_terra)
        print(f"Writing {names[fn]}...")
        utils.write_jsonl(f"outputs/{names[fn]}.jsonl", preds)
    shutil.make_archive("outputs", "zip", "outputs/")
