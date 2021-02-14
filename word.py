import json


if __name__ == '__main__':
    for split in ("val", "test"):
        data = list(map(json.loads, open(f"datasets/DaNetQA/{split}.jsonl")))
        words = [line["question"].strip().split(' ')[0] for line in data]

        bil = [word.startswith("Был") for word in words]
        li = ["ли" in line["question"] for line in data]
        est = [word.startswith("Есть") for word in words]
        pravda = [word.startswith("Правда") for word in words]

        feats = [tuple(int(y) for y in x) for x in zip(est, li, bil, pravda)]
        open(f"scores/word/word.{split}.scores", 'w').write('\n'.join(map(str, feats)))
