from zeronew import get_words
import json


def process(name):
    with open(f"datasets/DaNetQA/{name}.jsonl") as fin:
        json_list = list(fin)
    result = []
    for json_str in json_list:
        line = json.loads(json_str)
        text = f"{line['question']} [SEP] {get_words(line['passage'])}"
        result.append('\t'.join((text, str(line.get("label", 0)))))

    open(f"datasets/danetqa_{name}.tsv", 'w').write('\n'.join(result))


if __name__ == '__main__':
    for split in ("train", "val", "test"):
        print(split)
        process(split)
    print("done")
