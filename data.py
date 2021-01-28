import utils
from functools import partial


def read_jsonl(fn):
    return map(utils.Map, utils.load_jsonl(f"datasets/{fn}.jsonl"))


def read_data(fn, fun):
    result = {}
    for idx, sample in fun(read_jsonl(fn)):
        result[idx] = sample
    return result


def preprocess_lidirus(data):
    for sample in data:
        yield sample.idx, utils.Map(
            idx=sample.idx,
            premise=sample.sentence1,
            hypothesis=sample.sentence2,
            label=int(sample.label == "entailment"),
            misc=utils.Map(
                logic=sample.logic
            )
        )


def read_lidirus(split):
    return read_data("LiDiRus/LiDiRus", preprocess_lidirus)


def preprocess_parus(data):
    for sample in data:
        first = sample.idx*2 + 0, utils.Map(
            premise=sample.premise,
            hypothesis=sample.choice1,
            label=int(sample.label == 0)
        )
        second = sample.idx * 2 + 1, utils.Map(
            premise=sample.premise,
            hypothesis=sample.choice2,
            label=int(sample.label == 1)
        )
        for idx, val in [first, second]:
            if sample.question == "cause":
                val.premise, val.hypothesis = val.hypothesis, val.premise
            val.type = sample.type
            val.idx = idx
            yield idx, val


def read_parus(split):
    return read_data(f"PARus/{split}", preprocess_parus)


def preprocess_rcb(data):
    for sample in data:
        misc = utils.Map()
        for key in list(sample.keys()):
            if key not in ["idx", "premise", "hypothesis", "label"]:
                misc[key] = sample.pop(key)
        sample.misc = misc
        sample.label = ["contradiction", "neutral", "entailment"].index(sample.label) if "label" in sample else 0
        yield sample.idx, sample


def read_rcb(split):
    return read_data(f"RCB/{split}", preprocess_rcb)


def preprocess_muserc(data):
    for passage in data:
        text = passage["passage"]["text"]
        for query in passage["passage"]["questions"]:
            question = query["question"]
            for result in query["answers"]:
                answer = result["text"]
                label = result.get("label", 0)
                yield result["idx"], dict(
                    passage=text,
                    question=question,
                    answer=answer,
                    label=label
                )


def read_muserc(split):
    return read_data(f"MuSeRC/{split}", preprocess_muserc)


def preprocess_terra(data):
    for sample in data:
        sample.label = int(sample.label == "entailment")
        yield sample.idx, sample


def read_terra(split):
    return read_data(f"TERRa/{split}", preprocess_terra)


def preprocess_russe(data):
    for sample in data:
        misc = utils.Map()
        for key in list(sample.keys()):
            if key.startswith("start") or key.startswith("end") or key.startswith("gold"):
                misc[key] = sample.pop(key)
        sample.misc = misc
        sample.label = int(sample.label is True)
        yield sample.idx, sample


def read_russe(split):
    return read_data(f"RUSSE/{split}", preprocess_russe)


def preprocess_rwsd(data):
    for sample in data:
        sample.word1 = sample.target["span1_text"]
        sample.word2 = sample.target["span2_text"]
        sample.misc = sample.pop("target")
        yield sample.idx, sample


def read_rwsd(split):
    return read_data(f"RWSD/{split}", preprocess_rwsd)


def preprocess_danetqa(data):
    for sample in data:
        sample.label = int(sample.label is True)
        yield sample.idx, sample


def read_danetqa(split):
    return read_data(f"DaNetQA/{split}", preprocess_danetqa)


def preprocess_rucos(data, nli=False):
    idx = 0
    for sample in data:
        entities = sample.passage["entities"]
        candidates = list(set(sample.passage["text"][e["start"]:e["end"]] for e in entities))
        passage = sample.passage.pop("text")
        del sample.passage
        if nli:
            sample.premise = passage
        else:
            sample.passage = passage
        qas = list(sample.qas)
        sample.misc = utils.Map()
        sample.misc["candidates"] = candidates
        sample.misc["entities"] = entities
        sample.misc["qas"] = sample.pop("qas")
        for query in qas:
            question = query["query"]
            answer = query["answers"][0]["text"] if "answers" in query else None
            if nli:
                for candidate in candidates:
                    line = question.replace("@placeholder", candidate)
                    sample.hypothesis = line
                    sample.label = int(candidate == answer)
                    yield idx, sample
                    idx += 1
            else:
                sample.candidates = candidates
                sample.question = question
                sample.answer = answer
                yield idx, sample
                idx += 1


def read_rucos_nli(split):
    return read_data(f"RuCoS/{split}", partial(preprocess_rucos, nli=True))


def read_rucos(split):
    return read_data(f"RuCoS/{split}", partial(preprocess_rucos, nli=False))


if __name__ == '__main__':
    for fn in [read_lidirus, read_rcb, read_parus, read_muserc, read_terra, read_russe,
               read_rwsd, read_danetqa, read_rucos_nli, read_rucos]:
        for split in ["train", "test", "val"]:
            dct = fn(split)
            if split == "train":
                print(fn.__name__, next(iter(dct.values())))
