import utils
from functools import partial
import re
import pandas as pd


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


def preprocess_parus(data, nli=True):
    for sample in data:
        if nli:
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
            for idx, val in (first, second):
                if sample.question == "cause":
                    val.premise, val.hypothesis = val.hypothesis, val.premise
                val.question = sample.question
                val.idx = sample.idx
                yield idx, val
        else:
            sample.label = sample.get("label", 0)
            yield sample.idx, sample


def read_parus(split):
    return read_data(f"PARus/{split}", preprocess_parus)


def read_parus_nonnli(split):
    return read_data(f"PARus/{split}", partial(preprocess_parus, nli=False))


def preprocess_rcb(data):
    for sample in data:
        misc = utils.Map()
        for key in list(sample.keys()):
            if key not in ["idx", "premise", "hypothesis", "label"]:
                misc[key] = sample.pop(key)
        sample.misc = misc
        sample.label = ("contradiction", "neutral", "entailment").index(sample.label) if "label" in sample else 0
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
        sample.label = sample.get("label", 0)
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


def preprocess_rucos(data, nli=False, single=True, interpolate=True):
    idx = 0
    for sample in data:
        entities = sample.passage["entities"]
        candidates = list(set(sample.passage["text"][e["start"]:e["end"]] for e in entities))
        passage = sample.passage.pop("text")
        sample.passage = passage
        qas = list(sample.qas)
        sample.misc = utils.Map()
        sample.misc["candidates"] = candidates
        sample.misc["entities"] = entities
        sample.misc["qas"] = sample.pop("qas")
        for query in qas:
            question = query["query"]
            answers = set(answer["text"] for answer in query["answers"]) if "answers" in query else ['' if nli else candidates[0]]
            if nli:
                for candidate in candidates:
                    if interpolate:
                        line = question.replace("@placeholder", candidate)
                    else:
                        line = question
                        sample.answer = candidate
                    sample.hypothesis = line
                    sample.label = int(candidate in answers)
                    yield idx, sample
                    idx += 1
            else:
                if single:
                    answers = [next(iter(answers))]
                for answer in answers:
                    sample.candidates = candidates
                    sample.question = question
                    sample.answer = answer
                    yield idx, sample
                    idx += 1


def read_rucos_nli(split):
    return read_data(f"RuCoS/{split}", partial(preprocess_rucos, nli=True))


def read_rucos(split):
    return read_data(f"RuCoS/{split}", partial(preprocess_rucos, nli=False))


trans_table = dict([(ord(x), ord(y)) for x, y in zip("‘’´“”«»–-", '""\'""""--')])


def repl_quotes(string):
    string = string.translate(trans_table).strip()
    while string[0] == string[-1] == '"':
        string = string[1:-1]
    while '""' in string:
        string = string.replace('""', '"')
    return string



def repl_lines(string):
    return string.replace('\n', ' ')


def remove_highlight(string):
    return string.replace('@highlight', '')


numbers_re = re.compile(r"\(.?\d\d?.?\)")


def strip_numbers(text):
    matches = re.findall(numbers_re, text)
    if len(matches) < 5:
        return text
    return ' '.join(sentence.strip() for sentence in re.split(numbers_re, text))


diacritic = r'́'


def remove_diacritics(text):
    return text.replace(diacritic, '')


def preprocess_text(text):
    for fn in [lambda x: x, repl_quotes, repl_lines, remove_highlight, strip_numbers, remove_diacritics]:
        text = fn(text)
        while '  ' in text:
            text = text.replace('  ', ' ').strip()
    return text


def preprocess_sample(sample):
    return {key: value if key in ("idx", "label", "misc") or not isinstance(value, str)
            else preprocess_text(value)
            for key, value in sample.items()}


def preprocess_dataset(dataset, fun=preprocess_sample):
    return {key: fun(value) for key, value in dataset.items()}


sort_order = ("question", "answer", "word", "word1", "word2", "text",
              "sentence1", "sentence2", "premise", "hypothesis", "passage")


def preprocess_bert(sample, fn):
    label = sample["label"]
    sample = {key: value for key, value in sample.items() if key not in ("idx", "misc", "label")}
    fragments = []
    for key in sorted(sample.keys(), key=lambda x: sort_order.index(x)):
        fragments.append(f"{key}: {sample[key]}")
    text = ' '.join(fragments)
    dataset_name = fn.__name__.split('_')[1]
    text = f"{dataset_name} {text}"
    return text, label


if __name__ == '__main__':
    splits = {}
    train = []
    val = []
    for fn in (read_lidirus, read_rcb, read_parus,  # read_parus_nonnli,
               read_muserc, read_terra, read_russe, read_rwsd, read_danetqa,  # read_rucos_nli,  # read_rucos
               ):
        for split in ("train", "test", "val"):
            data = fn(split)
            data = preprocess_dataset(data)
            data = preprocess_dataset(data, fun=partial(preprocess_bert, fn=fn))
            dct = next(iter(data.values()))
            if isinstance(dct, dict) and "misc" in dct:
                del dct["misc"]
            if split == "train":
                print(fn.__name__, dct)
            if split not in splits:
                splits[split] = []
            splits[split] += data.values()

    for name, df in splits.items():
        pd.DataFrame(df).to_csv(f"datasets/{name}.csv", header=False, index=False)
