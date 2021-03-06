import re
import json
from nltk.tokenize import RegexpTokenizer
from DeBERTa.DeBERTa import deberta
import torch
from tqdm import tqdm


tokenizer = RegexpTokenizer(r'\d+[ ]+\d+[ ]+\d+|\d+[ ]+\d+|[a-zA-Z]+[.]+[a-zA-Z]+|[A-Z]+[a-z]+|\d+[.,:+-]+\d+|\w+')


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
    # torch.set_num_threads(8)
    torch.set_grad_enabled(False)
    print("Loading model...")
    model = deberta.DeBERTa(pre_trained="xlarge-v2-mnli")
    model.apply_state()
    print("Loading tokenizer...")
    vocab_path, vocab_type = deberta.load_vocab(pretrained_id="xlarge-v2-mnli")
    tokenizer = deberta.tokenizers[vocab_type](vocab_path)

    def make_preds_zero_shot(name):
        json_list = list(open(f"datasets/DaNetQA/{name}.jsonl"))
        table = json.load(open("translations/translation.json"))

        for i, json_str in enumerate(tqdm(json_list)):
            result = json.loads(json_str)

            premise_tokens = tokenizer.tokenize(get_words(table[result["passage"].strip()]))
            hypothesis_tokens = tokenizer.tokenize(table[result["question"].strip()] + " Yes.")
            all_tokens = ['[CLS]'] + premise_tokens + ['[SEP]'] + hypothesis_tokens + ['[SEP]']
            input_ids = tokenizer.convert_tokens_to_ids(all_tokens[:512])
            state = model(torch.LongTensor([input_ids]))[-1][0]

            torch.save(state, f"states/platinum/xlarge/{name}/{result['idx']}.pt")

    for split in ("train", "val", "test"):
        print(f"Processing {split}...")
        make_preds_zero_shot(split)


if __name__ == '__main__':
    main()
