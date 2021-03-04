import pandas as pd
import json
import re


if __name__ == '__main__':
    df = pd.read_pickle("datasets/boolq-record-cb-copa-multirc-rte_train.pkl")
    en = []
    for line in map(json.loads, open("datasets/BoolQ/train.jsonl")):
        en.append((f"boolq question: {line['question']} passage: {line['passage']}", int(line["answer"])))
    pattern = re.compile(r"<b>Sent \d\d?: </b>")
    for paragraph in json.load(open("datasets/MultiRC/train.json"))["data"]:
        paragraph = paragraph["paragraph"]
        text = paragraph["text"]
        text = ''.join(re.split(pattern, text.replace('<br>', ' '))).strip()
        for quest in paragraph["questions"]:
            question = quest["question"]
            for ans in quest["answers"]:
                answer = ans["text"]
                label = int(ans["isAnswer"])
                en.append((f"multirc question: {question} answer: {answer} passage: {text}", label))
    df = df.append(pd.DataFrame(en, columns=df.columns)).reset_index()
    df.to_pickle("datasets/four_train.pkl", protocol=4)
