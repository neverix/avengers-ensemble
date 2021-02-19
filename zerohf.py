import re
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import data
import torch
from torch.nn import functional as F
import json
from DeBERTa.DeBERTa import deberta
from DeBERTa.DeBERTa.deberta.cache_utils import load_model_state
from DeBERTa.DeBERTa.apps.models.sequence_classification import SequenceClassificationModel


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
name = "zerode/delx"


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
    for k, v in sorted(datas.items()):
        # torch.cuda.empty_cache()
        premise_tokens = tokenizer.tokenize(get_words(table[v["passage"].strip()]))
        hypothesis_tokens = tokenizer.tokenize(table[v["question"].strip()] + " Yes.")
        all_tokens = ['[CLS]'] + premise_tokens + ['[SEP]'] + hypothesis_tokens + ['[SEP]']
        input_ids = tokenizer.convert_tokens_to_ids(all_tokens[:512])
        logits, _ = model(torch.LongTensor([input_ids]))
        yield tuple(float(x) for x in logits.flatten())


def main():
    torch.set_grad_enabled(False)
    model_name = "xlarge-v2-mnli"
    print("Loading model...")
    _, model_config = load_model_state(model_name)
    model = SequenceClassificationModel(model_config, num_labels=3, drop_out=0.0, pre_trained=model_name)
    # model.apply_state()
    print("Loading tokenizer...")
    vocab_path, vocab_type = deberta.load_vocab(pretrained_id=model_name)
    tokenizer = deberta.tokenizers[vocab_type](vocab_path)
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
