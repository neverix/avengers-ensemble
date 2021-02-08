import json
import re
from copy import deepcopy


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


def it_muserc(it, prep=False, obj=False, repl=False):
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
    for (*_, ans), label in zip(it_muserc(jsonl, obj=True), it):
        ans["label"] = label
    return jsonl
