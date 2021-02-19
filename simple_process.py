from zeronew import get_words
import json
import pandas as pd
import translator
from tqdm import tqdm
import ast
translation = json.load(open("translations/translation.json"))


def process_danetqa(name):
    with open(f"datasets/DaNetQA/{name}.jsonl") as fin:
        json_list = list(fin)
    result = []
    for json_str in json_list:
        line = json.loads(json_str)
        text = f"question: {line['question']} passage: {get_words(line['passage'])}"
        result.append((text, str(int(ast.literal_eval(line.get("label", '0'))))))
    df = pd.DataFrame(result)
    df.to_pickle(f"datasets/danetqa_{name}.pkl", protocol=4)


def process_muserc(name):
    global translation
    with open(f"datasets/MuSeRC/{name}.jsonl") as fin:
        json_list = list(map(json.loads, fin))
    result = []
    for passage in tqdm(json_list):
        passage = passage["passage"]
        text = passage["text"].strip()
        if text not in translation:
            translator.translate_all([text], "translations/translation.json")
            translation = json.load(open("translations/translation.json"))
        text = translation[text]
        questions = passage["questions"]
        for question in questions:
            q = translation[question["question"].strip()]
            for ans in question["answers"]:
                answer = translation[ans["text"].strip()]
                txt = f"question: {q} answer: {answer} passage: {text}"
                result.append((txt, str(ans.get("label", 0))))
    df = pd.DataFrame(result)
    df.to_pickle(f"datasets/muserc_{name}.pkl", protocol=4)


if __name__ == '__main__':
    for split in ("train", "val", "test"):
        print(split)
        # process_muserc(split)
        process_danetqa(split)
    print("done")
