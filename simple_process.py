from zeronew import get_words
import json
import pandas as pd


def process_danetqa(name):
    with open(f"datasets/DaNetQA/{name}.jsonl") as fin:
        json_list = list(fin)
    result = []
    for json_str in json_list:
        line = json.loads(json_str)
        text = f"question: {line['question']} passage: {get_words(line['passage'])}"
        result.append('\t'.join((text, str(line.get("label", 0)))))

    open(f"datasets/danetqa_{name}.tsv", 'w').write('\n'.join(result))


def process_muserc(name):
    with open(f"datasets/MuSeRC/{name}.jsonl") as fin:
        json_list = list(map(json.loads, fin))
    result = []
    for passage in json_list:
        passage = passage["passage"]
        text = passage["text"]
        questions = passage["questions"]
        for question in questions:
            q = question["question"]
            for answer in question["answers"]:
                txt = f"question: {q} answer: {answer['text']} passage: {text}"
                result.append((txt, str(answer.get("label", 0))))
    df = pd.DataFrame(result)
    df.to_pickle(f"datasets/muserc_{name}.pkl", protocol=4)


if __name__ == '__main__':
    for split in ("train", "val", "test"):
        print(split)
        process_muserc(split)
    print("done")
