import re
import json
from nltk.tokenize import RegexpTokenizer
from transformers import pipeline


tokenizer = RegexpTokenizer(r'\d+[ ]+\d+[ ]+\d+|\d+[ ]+\d+|[a-zA-Z]+[.]+[a-zA-Z]+|[A-Z]+[a-z]+|\d+[.,:+-]+\d+|\w+')


def preprocess_word(word):
    word = word.replace(' ', '')
    word = word.replace('.', ',')
    # word = word.replace('-', ':')

    return word


trans_table = dict([(ord(x), ord(y)) for x, y in zip("‘’´“”«»—–-", '""\'""""---')])


def get_words(text):
    text = text.translate(trans_table).strip()
    while '  ' in text:
        text = text.replace('  ', ' ')
    return text
    # text = re.sub('[(]+[\dЗ\W]+[)]', '', text)
    words = [preprocess_word(word) for word in tokenizer.tokenize(text) if word]
    return ' '.join(words)


def main():
    classifier = pipeline("zero-shot-classification", model="ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli", device=0)

    def make_preds_zero_shot(name):
        preds = []

        json_list = list(open(f"datasets/DaNetQA/{name}.jsonl"))
        table = json.load(open("translations/translation.json"))

        for json_str in json_list:
            result = json.loads(json_str)

            cl_preds = classifier(get_words(table[result["passage"].strip()]), [''],
                                  multi_class=False, hypothesis_template=f"{table[result['question'].strip()]} Yes." + '{}')
            preds.append(cl_preds['scores'][0])

        return preds

    for split in ("val", "test"):
        print(f"Processing {split}...")
        preds = make_preds_zero_shot(split)
        print("Writing...")
        open(f"scores/qa/en-albzero.{split}.scores", 'w').write('\n'.join(map(str, preds)))
        # print()
        # exit()


if __name__ == '__main__':
    main()
