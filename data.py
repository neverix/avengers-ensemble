import utils
import translator
from functools import partial
import re
import pandas as pd
import csv


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


trans_table = dict([(ord(x), ord(y)) for x, y in zip("‘’´“”«»—–-", '""\'""""---')])


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


def remove_diacritics(text):
    return text.replace(r'́', '').replace('\xad', '')


def preprocess_text(text):
    for fn in [lambda x: x, remove_highlight, repl_quotes, repl_lines, strip_numbers, remove_diacritics]:
        text = fn(text).strip()
        while '  ' in text:
            text = text.replace('  ', ' ').strip()
    return text.strip()


def preprocess_sample(sample):
    return {key: value if key in ("idx", "label", "misc") or not isinstance(value, str)
            else preprocess_text(value)
            for key, value in sample.items()}


def preprocess_dataset(dataset, fun=preprocess_sample):
    return {key: fun(value) for key, value in dataset.items()}


sort_order = ("question", "answer", "word", "word1", "word2", "text",
              "sentence1", "sentence2", "premise", "hypothesis", "passage")
replacements = dict(
    question="вопрос", answer="ответ", word="слово", word1="слово1", word2="слово2", text="текст",
    sentence1="предложение1", sentence2="предложение2", premise="предложение", hypothesis="гипотеза", passage="текст",
)
do_replace = False
is_upper = False
mapping = {
    read_danetqa: "boolq",
    read_rucos: "record",
    read_rcb: "cb",
    read_parus: "copa",
    read_muserc: "multirc",
    read_terra: "rte"
}


def fn_name(fn):
    if fn in mapping:
        return mapping[fn]
    return fn.__name__.split('_')[1]


def preprocess_bert(sample, fn, single=False):
    label = sample["label"]
    sample = {key: value for key, value in sample.items() if key not in ("idx", "misc", "label")}
    fragments = []
    order = sort_order
    if fn == read_danetqa:
        order = order  # list(reversed(order))
    for key in sorted(sample.keys(), key=lambda x: order.index(x)):
        name = key
        if do_replace:
            name = replacements[name]
        if is_upper:
            name = name[0].upper() + name[1:]
        fragments.append(f"{name}: {sample[key]}")
    text = ' '.join(fragments)
    text = f"{'' if single and False else (fn_name(fn) + ' ')}{text}"
    return text, label


def replace_table(sample, table=None):
    if table is None:
        table = {}
    return {key: (table[value.strip()] if isinstance(key, str) and key not in ("idx", "misc", "label") else value)
            for key, value in sample.items()}


def to_translate(data):
    result = set()
    for sample in data.values():
        result.update({value for key, value in sample.items() if isinstance(value, str) and key not in ("idx", "label", "misc")})
    return result


data_funs = (read_lidirus, read_rcb, read_parus,  # read_parus_nonnli,
             read_muserc, read_terra, read_russe, read_rwsd, read_danetqa,  # read_rucos_nli,  # read_rucos
             )
translation_path = "translations/translation.json"
dont_process = (read_danetqa, read_muserc)


def load_all(tasks=data_funs, verbose=False, translate=False):
    splits = {}
    source = {}
    for fn in data_funs:
        for split in ("train", "test", "val"):
            print("Reading", fn.__name__, split)
            if split not in splits:
                splits[split] = []
            if fn not in tasks:
                # splits[split] += [('0', 0) for _ in src]
                continue
            src = fn(split)
            if fn not in dont_process:
                dataset = preprocess_dataset(src)
            else:
                dataset = src
            if translate:
                table = translator.translate_all(to_translate(dataset), translation_path)
                dataset = preprocess_dataset(dataset, fun=partial(replace_table, table=table))
            data = preprocess_dataset(dataset, fun=partial(preprocess_bert, fn=fn, single=len(tasks) == 1))
            source[(fn, split)] = src, dataset, data
            dct = next(iter(data.values()))
            if isinstance(dct, dict) and "misc" in dct:
                del dct["misc"]
            if verbose and split == "val":
                print(fn.__name__, dct)
            splits[split] += [v for k, v in sorted(data.items())]
    return splits, source


def make_df(tasks, is_tsv=False, is_pkl=False, source_only=False, **kwargs):
    tsv_params = dict(sep="\t", quoting=csv.QUOTE_NONE)
    print("Preprocessing", tasks)
    splits, source = load_all(tasks, verbose=True, **kwargs)
    for name, df in splits.items():
        file_name = '' if set(tasks) == set(data_funs) else '-'.join(fn_name(task) for task in tasks) + '_'
        df = pd.DataFrame(df)
        if source_only:
            df.drop(columns=[df.columns[-1]], inplace=True)
        name = f"datasets/{file_name}{name}.{'pkl' if is_pkl else 'txt' if source_only else 'tsv' if is_tsv else 'csv'}"
        if is_pkl:
            df.to_pickle(name, protocol=4)
        else:
            df.to_csv(name, header=False, index=False, **(tsv_params if is_tsv else {}))


if __name__ == '__main__':
    make_df([read_danetqa, read_muserc, read_terra], source_only=True, is_tsv=True, translate=True)
    exit()
    make_df([read_danetqa], is_pkl=True, translate=True)
    datas = [read_danetqa, read_rucos, read_rcb, read_parus, read_muserc, read_terra]
    make_df(datas, is_tsv=True)
    make_df(datas, source_only=True, is_tsv=True)
    make_df([read_muserc, read_danetqa], is_pkl=True, translate=True)
    exit()
    make_df([read_muserc], is_pkl=True, translate=True)
    make_df([read_muserc], source_only=True, is_pkl=True, translate=True)
    exit()
    make_df([read_danetqa], is_tsv=True)
    make_df([read_danetqa], source_only=True, is_tsv=True)
    datas = [read_danetqa, read_rucos, read_rcb, read_parus, read_muserc, read_terra]
    make_df(datas, is_tsv=True, translate=True)
    make_df(datas, source_only=True, is_tsv=True, translate=True)
    exit()
    load_all(data_funs, verbose=True, translate=True)
    exit()
    # print(list(read_split("mbert/mbert", "test")[read_rcb].values())[:10])
    make_df([read_rcb])
    make_df([read_terra])
    make_df(data_funs)
