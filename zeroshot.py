import re
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from transformers import pipeline
import data


def process_terra(sample):
    return get_words(sample["premise"]), "{} следует из текста", (get_words(sample["hypothesis"]),)


tokenizer = RegexpTokenizer(r'\d+[ ]+\d+[ ]+\d+|\d+[ ]+\d+|[a-zA-Z]+[.]+[a-zA-Z]+|[A-Z]+[a-z]+|\d+[.,:+-]+\d+|\w+')
processors = {
    data.read_terra: process_terra
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


def make_preds_zero_shot(model, dataset, split):
    datas = dataset(split)
    datas = data.preprocess_dataset(datas)
    processor = processors[dataset]
    for k, v in sorted(datas.items()):
        premise, hypothesis_template, hypotheses = processor(v)
        preds = model(premise, hypotheses, multi_class=False, hypothesis_template=hypothesis_template)
        yield tuple(preds["scores"])


def main():
    print("loading model")
    classifier = pipeline("zero-shot-classification", model="vicgalle/xlm-roberta-large-xnli-anli", device=0)
    print("model loaded")
    for split in ("val", "test"):
        print("processing", split)
        results = []
        for dataset in processors:
            print(" computing", dataset.__name__)
            preds = make_preds_zero_shot(classifier, dataset, split)
            for pred in tqdm(preds, total=len(dataset(split))):
                results.append(pred)
        print(" writing")
        open(f"scores/zero-norm/zero.{split}.scores", 'w').write('\n'.join(str(p) for p in results))


if __name__ == '__main__':
    main()
