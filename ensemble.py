import data


models = ("mbert/mbert",)
processors = {

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
            results = {key: {} for key in data1}
            for model in models:
                result = preds[(model, split)][fn]
                for key in result:
                    for i, e in enumerate(result[key]):
                        feat_name = f"{model}_{i}"
                        results[key][feat_name] = e
            splits[split] = results
        total[fn] = splits
    return total






if __name__ == '__main__':
    dataset = data.load_all(verbose=True)
    feats = make_feats(dataset)[data.read_terra]["val"]
    print(feats)
