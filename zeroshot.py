import os
import re
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from transformers import pipeline, tokenization_utils_base
import data


def process_terra(sample):
    return get_words(sample["premise"]), "{} следует из текста", (get_words(sample["hypothesis"]),)


def process_lidirus(sample):
    return get_words(sample["premise"]), "{} следует из текста", (get_words(sample["hypothesis"]),)


def process_rcb(sample):
    return get_words(sample["premise"]), "{}", (f"{get_words(sample['hypothesis'])} не следует из текста",
                                                f"{get_words(sample['hypothesis'])} не относится к тексту",
                                                f"{get_words(sample['hypothesis'])} следует из текста",)


def process_parus(sample):
    return get_words(sample["premise"]), "{} следует из текста", (get_words(sample["hypothesis"]),)


def process_russe(sample):
    return (get_words(sample["sentence1"]), f" - слово {sample['word']} применяется в том же смысле {'{}'} - ",\
            (get_words(sample["sentence2"]),)


def process_danetqa(sample):
    return get_words(sample["passage"]), f"{sample['question']} Да." + '{}', ['']


def process_muserc(sample):
    return get_words(sample["passage"]), "Ответ на вопрос " + sample["question"] + " {}.", (sample["answer"],)


tokenizer = RegexpTokenizer(r'\d+[ ]+\d+[ ]+\d+|\d+[ ]+\d+|[a-zA-Z]+[.]+[a-zA-Z]+|[A-Z]+[a-z]+|\d+[.,:+-]+\d+|\w+')
processors = {
    # data.read_danetqa: process_danetqa,
    # data.read_terra: process_terra,
    # data.read_lidirus: process_lidirus,
    # data.read_rcb: process_rcb,
    data.read_russe: process_russe,
    # data.read_muserc: process_muserc,
    # data.read_parus: process_parus,
}
name = "how-about-we-name-this-normally-for-once/mdeberta-v3-base-xnli-test1"  # "zero-norm/super-plus"


def preprocess_word(word):
    word = word.replace(' ', '')
    word = word.replace('.', ',')
    word = word.replace('-', ':')

    return word


def get_words(text):
    text = re.sub('[(]+[\dЗ\W]+[)]', '', text)
    words = [preprocess_word(word) for word in tokenizer.tokenize(text)]
    return ' '.join(words)

def make_dataset_zero_shot(dataset, split):
    datas = dataset(split)
    # datas = data.preprocess_dataset(datas)
    processor = processors[dataset]
    for k, v in sorted(datas.items()):
        premise, hypothesis_template, hypotheses = processor(v)
        yield premise, hypothesis_template, hypotheses


def make_preds_zero_shot(model, dataset, split):
    for premise, hypothesis_template, hypotheses in make_dataset_zero_shot(dataset, split):
        preds = model(premise, hypotheses, multi_label=False, hypothesis_template=hypothesis_template)
        yield tuple(preds["scores"])


def main():
    print("loading model")
    # classifier = pipeline("zero-shot-classification", model="vicgalle/xlm-roberta-large-xnli-anli", device=0)
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", device=0)
    print("model loaded")
    for split in ("val", "test"):
        print("processing", split)
        results = []
        for dataset in data.data_funs:
            if dataset in processors:
                print(" computing", dataset.__name__)
                preds = make_preds_zero_shot(classifier, dataset, split)
                for pred in tqdm(preds, total=len(dataset(split))):
                    results.append(pred)
        print(" writing")
        os.makedirs(f"scores/{os.path.dirname(name)}", exist_ok=True)
        open(f"scores/{name}.{split}.scores", 'w').write('\n'.join(str(p) for p in results))


if __name__ == '__main__':
    main()
