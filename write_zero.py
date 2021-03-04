import json
import re
import pandas as pd


def repl_lines(string):
    return string.replace('\n', ' ')


def reregex(string):
    for char in '.?!"\'':
        string = string.replace(char + ' @highlight', char)
    return string.replace(' @highlight ', '. ')
    print(string)
    string = re.sub(string,
                  r"(\.|\?|\!|\"|\')\n@highlight\n",
                  r"\1 ")
    print(string)
    exit()
    return re.sub(string,r'\n@highlight\n', '. ')


def preprocess_text(text, p=False):
    for fn in [# lambda x: x.replace('@Placeholder', '@placeholder'),
        (reregex if p else (lambda x: x)),
               # remove_highlight,
               # repl_quotes,
               repl_lines,
               # strip_numbers,
               # remove_diacritics,
               # remove_bracket
               ]:
        text = fn(text).strip()
        while '  ' in text:
            text = text.replace('  ', ' ').strip()
    return text.strip()


def preprocess_data(split, over=False, ren=True, start=0, train=False, simple=False, super=False, demove=False):
    if train and not demove:
        data = json.load(open("datasets/record/train.json"))["data"]
    else:
        data = list(map(json.loads, open(f"datasets/RuCoS/{split}.jsonl")))
    if train and not demove:
        prep = lambda x: preprocess_text(x)
        post = lambda x: preprocess_text(x, True)
    else:
        table_json = open("translations/translation.json").read()
        table = json.loads(table_json)
        prep = lambda x: table[preprocess_text(x)]
        post = lambda x: preprocess_text(x, True)
    proc = []
    i = 0
    for par in data:
        text = par["passage"]["text"]
        entities = []
        for ent in par["passage"]["entities"]:
            ent = text[ent["start"]:ent["end"]]
            if ent not in entities:
                entities.append(ent)
        for qa in par["qas"]:
            try:
                query = qa["query"]
                ref = f"record query: {prep(query)} entities: {', '.join(map(prep, entities))} passage: {post(prep(text))}"
                correct_answers = []
                if "answers" in qa:
                    for ans in qa["answers"]:
                        ans = text[ans["start"]:ans["end"]]
                        if ans not in correct_answers:
                            correct_answers.append(ans)
                correct = [ent for ent in entities if ent in correct_answers]
                incorrect = [ent for ent in entities if ent not in correct_answers]
                i += len(entities)
                if simple:
                    yield ref, ', '.join(map(prep, entities))
                elif super:
                    for x in correct:
                        yield ref, prep(x)
                elif train:
                    correct *= len(incorrect) // max(len(correct), 1)
                    for x in correct:
                        yield ref.replace("@placeholder", prep(x)), 1
                    for x in incorrect:
                        yield ref.replace("@placeholder", prep(x)), 0
                else:
                    for ent in entities:
                        yield ref, prep(ent)
            except KeyError:
                pass
'''
for split in ("val", "test"):
    print(split)
    x, y = map(list, zip(*preprocess_data(split)))
    print(len(x))
    open(f"datasets/rucos-{split}.xn.txt", 'w').write('\n'.join(x))
    open(f"datasets/rucos-{split}.yn.txt", 'w').write('\n'.join(y))
exit()
'''

print('ru')
df = pd.DataFrame(list(preprocess_data("train", train=True, demove=True))).sample(100_000).reset_index(drop=True)
print(df)
print('write')
df.to_pickle("datasets/rucos-train-ru-zero.pkl", protocol=4)
exit()


print("train-ru")
df = pd.DataFrame(list(preprocess_data("train", super=True)))
print(df)
df.to_csv("datasets/rucos-train-ru-alli.tsv", sep='\t', header=None, index=None)
exit()

print("train")
df = pd.DataFrame(list(preprocess_data("train", train=True)))
df.to_pickle("datasets/rucos-train.pkl")

for split in ("val", "test"):
    print(split)
    x, y = map(list, zip(*preprocess_data(split)))
    open(f"datasets/rucos-{split}.x.txt", 'w').write('\n'.join(x))
    open(f"datasets/rucos-{split}.y.txt", 'w').write('\n'.join(y))

for split in ("val", "test"):
    print(split)
    x, y = map(list, zip(*preprocess_data(split, simple=True)))
    open(f"datasets/rucos-{split}.a.txt", 'w').write('\n'.join(x))
    open(f"datasets/rucos-{split}.c.txt", 'w').write('\n'.join(y))
