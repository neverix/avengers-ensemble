import re
import json
from nltk.tokenize import RegexpTokenizer
from transformers import pipeline
from DeBERTa.DeBERTa import deberta


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
    model = deberta.DeBERTa(pre_trained="base-mnli")
    vocab_path, vocab_type = deberta.load_vocab(pretrained_id="large-mnli")
    tokenizer = deberta.tokenizers[vocab_type](vocab_path)
    exit()

    def make_preds_zero_shot(name):
        preds = []

        json_list = list(open(f"datasets/DaNetQA/{name}.jsonl"))
        table = json.load(open("translations/translation.json"))

        for json_str in json_list:
            result = json.loads(json_str)

            premise_tokens = tokenizer.tokenize(get_words(table[result["passage"].strip()]))
            hypothesis_tokens = tokenizer.tokenize(table[result["question"].strip()] + " Yes.")
            all_tokens = ['[CLS]'] + premise_tokens + ['[SEP]'] + hypothesis_tokens + ['[SEP]']
            input_ids = tokenizer.convert_tokens_to_ids(all_tokens[:512])

            # cl_preds = classifier(, [''],
            #                       multi_class=False, hypothesis_template=f"{table[result['question'].strip()]} Yes." + '{}')
            preds.append(0['scores'][0])

        return preds

    for split in ("val", "test"):
        print(f"Processing {split}...")
        preds = make_preds_zero_shot(split)
        print("Writing...")
        open(f"scores/qa/en-azero.{split}.scores", 'w').write('\n'.join(map(str, preds)))
        # print()
        # exit()


if __name__ == '__main__':
    main()
