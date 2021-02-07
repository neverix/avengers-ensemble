import re
import json
from nltk.tokenize import RegexpTokenizer
from transformers import pipeline


tokenizer = RegexpTokenizer(r'\d+[ ]+\d+[ ]+\d+|\d+[ ]+\d+|[a-zA-Z]+[.]+[a-zA-Z]+|[A-Z]+[a-z]+|\d+[.,:+-]+\d+|\w+')
templates = {
    "TERRa": ('{} следует из текста', ('неверно', 'верно'))
}


def preprocess_word(word):
    word = word.replace(' ', '')
    word = word.replace('.', ',')
    word = word.replace('-', ':')

    return word


def get_words(text):
    text = re.sub('[(]+[\dЗ\W]+[)]', '', text)
    words = [preprocess_word(word) for word in tokenizer.tokenize(text)]
    return ' '.join(words)


def main():
    classifier = pipeline("zero-shot-classification", model="vicgalle/xlm-roberta-large-xnli-anli", device=0)

    def make_preds_zero_shot(dataset, name):
        preds = []

        with open(f"datasets/{dataset}/{name}.jsonl") as fin:
            json_list = list(fin)

        for json_str in json_list:
            result = json.loads(json_str)

            premise = get_words(result['premise'])
            hypothesis = get_words(result['hypothesis'])

            cl_preds = classifier(premise, [hypothesis], multi_class=False, hypothesis_template=hypothesis_template)
            preds.append(cl_preds['scores'][0])

        return preds
    for dataset in templates:
        for split in ("val", "test"):
            print(f" Processing {split}...")
            preds = make_preds_zero_shot(dataset, split)
            print(" Writing...")
            open(f"scores/zero/zero-new.{split}.scores", 'w').write('\n'.join(f"({p},)" for p in preds))
            # print()
            # exit()


if __name__ == '__main__':
    main()
