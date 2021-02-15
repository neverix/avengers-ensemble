import re
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import data
import torch
from torch.nn import functional as F
import json


def process_terra(sample):
    return get_words(sample["premise"]), "{} следует из текста", (get_words(sample["hypothesis"]),)


def process_lidirus(sample):
    return get_words(sample["premise"]), "{} следует из текста", (get_words(sample["hypothesis"]),)


def process_rcb(sample):
    return get_words(sample["premise"]), "{}", (f"{get_words(sample['hypothesis'])} не следует из текста",
                                                f"{get_words(sample['hypothesis'])} не относится к тексту",
                                                f"{get_words(sample['hypothesis'])} следует из текста",)


def process_russe(sample):
    return get_words(sample["sentence1"]), f"слово {sample['word']} применяется в том же смысле {'{}'}",\
           (get_words(sample["sentence2"]),)


def process_danetqa(sample):
    return get_words(sample["passage"]), f"{sample['question']} Да." + '{}', ['']


def process_muserc(sample):
    return get_words(sample["passage"]), "Ответ на вопрос " + sample["question"] + " {}", (sample["answer"],)


tokenizer = RegexpTokenizer(r'\d+[ ]+\d+[ ]+\d+|\d+[ ]+\d+|[a-zA-Z]+[.]+[a-zA-Z]+|[A-Z]+[a-z]+|\d+[.,:+-]+\d+|\w+')
processors = {
    data.read_danetqa: process_danetqa,
    # data.read_terra: process_terra,
    # data.read_lidirus: process_lidirus,
    # data.read_rcb: process_rcb,
    # data.read_russe: process_russe,
    # data.read_muserc: process_muserc,
}
name = "zerode/en"


def preprocess_word(word):
    word = word.replace(' ', '')
    word = word.replace('.', ',')
    word = word.replace('-', ':')

    return word


def get_words(text):
    text = re.sub('[(]+[\dЗ\W]+[)]', '', text)
    words = [preprocess_word(word) for word in tokenizer.tokenize(text)]
    return ' '.join(words)


def make_preds_zero_shot(model, tokenizer, dataset, table, split):
    datas = dataset(split)
    # datas = data.preprocess_dataset(datas)
    processor = processors[dataset]
    cls = torch.LongTensor([[tokenizer.cls_token_id]])
    sep = torch.LongTensor([[tokenizer.sep_token_id]])
    tokenize = lambda x: tokenizer(x, add_special_tokens=False, return_tensors='pt', max_length=512, truncation=True).input_ids
    for k, v in sorted(datas.items()):
        # torch.cuda.empty_cache()
        passage = get_words(table[v["passage"].strip()])
        hypothesis = table[v["question"].strip()] + " Yes."
        tokens = torch.cat((cls, tokenize(passage), sep, tokenize(hypothesis), sep), dim=1)
        tokens = tokens.to(model.device)
        yield tuple(float(x) for x in F.softmax(model(tokens).logits.flatten()))


def main():
    torch.set_grad_enabled(False)
    print("loading model")
    model_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).cuda()
    print("model loaded")
    table = json.load(open("translations/translation.json"))
    for split in ("val", "test"):
        print("processing", split)
        results = []
        for dataset in data.data_funs:
            if dataset in processors:
                print(" computing", dataset.__name__)
                preds = make_preds_zero_shot(model, tokenizer, dataset, table, split)
                for pred in tqdm(preds, total=len(dataset(split))):
                    results.append(pred)
        print(" writing")
        open(f"scores/{name}.{split}.scores", 'w').write('\n'.join(str(p) for p in results))


if __name__ == '__main__':
    main()
