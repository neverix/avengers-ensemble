import pandas as pd
import json
import re
import pandas as pd
from tqdm import tqdm


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
    for fn in [(reregex if p else (lambda x: x)),
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
print("loading trans")
import cfuzzyset
table_json = open("translations/translation.json").read()
print("decoding...")
table = json.loads(table_json)
prep = lambda x: table[preprocess_text(x)]
post = lambda x: preprocess_text(x, True)
print("the data")
data = list(map(json.loads, open(f"datasets/RuCoS/train.jsonl")))
print("doing old")
i = 0
res = {}
pre = preprocess_text
fs = cfuzzyset.cFuzzySet()
for par in tqdm(data):
    text = par["passage"]["text"]
    entities = []
    for ent in par["passage"]["entities"]:
        ent = text[ent["start"]:ent["end"]]
        if ent not in entities:
            entities.append(ent)
    for qa in par["qas"]:
        key = ', '.join(prep(ent) for ent in entities)
        fs.add(key)
        try:
            query = qa["query"]
            res[key] = pre(query), [pre(ent) for ent in entities], pre(text)
            '''ref = f"record query: {prep(query)} entities: {', '.join(map(prep, entities))} passage: {post(prep(text))}"
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
                correct *= len(incorrect) // len(correct)
                for x in correct:
                    yield ref.replace("@placeholder", x), 1
                for x in incorrect:
                    yield ref.replace("@placeholder", x), 0
            else:
                for ent in entities:
                    yield ref, prep(ent)'''
        except ZeroDivisionError:
            res.append((len(entities), None))


print("loading tsv")
backup = pd.read_csv("translations/translation-backup.tsv", sep='\t', header=None)
tups = [tuple(y) for x, y in backup.iterrows()]
last_placeholder = ''
num_last = -1
result = []
ents = []
print("run tups")
for a, b in tups:
    a = a.split('record query: ')[1]
    try:
        query, *a = a.split(' entities: ')
        a = ' entities: '.join(a)
    except ValueError:
        print(a)
        exit()
    entities, passage = a.split(' passage: ')
    entities = entities.split(', ')

    if query != last_placeholder:
        result.append((last_placeholder, entities, passage, num_last))
        num_last = 1
        last_placeholder = query
    else:
        num_last += 1
print("rerite")
result.append((last_placeholder, entities, passage, num_last))
new_trans = []
for last, entities, text, _ in tqdm(result[1:]):
    #print(last in table.values())
    key = ', '.join(entities)
    try:
        sq, se, st = res[key]
    except KeyError:
        continue
        sq, se, st = res[fs.get(key)[0][1]]
    add = [(sq, last), (st, text)] + list(zip(se, entities))
    # print(add)
    new_trans += add
new_trans = dict(new_trans)
print(len(table))
for key in new_trans:
    del table[key]
# table.update(new_trans)
print(len(new_trans), len(table))


json.dump(table, open("translations/translation-with"
                      "out.json", 'w'), ensure_ascii=False)

exit()
lens = {}
queries = {}
for idx, (length, q) in enumerate(res):
    if length not in lens:
        lens[length] = set()
    lens[length].add(idx)
    if q is not None:
        queries[q] = idx


print("loading tsv")
backup = pd.read_csv("translations/translation-backup.tsv", sep='\t', header=None)
tups = [tuple(y) for x, y in backup.iterrows()]
last_placeholder = ''
num_last = -1
result = []
print("run tups")
for a, b in tups:
    a = a.split('record query: ')[1]
    try:
        query, *a = a.split(' entities: ')
        a = ' entities: '.join(a)
    except ValueError:
        print(a)
        exit()
    entities, passage = a.split(' passage: ')
    entities = entities.split(', ')

    if query != last_placeholder:
        result.append((last_placeholder, entities, passage, num_last))
        num_last = 1
        last_placeholder = query
    else:
        num_last += 1
result.append((last_placeholder, entities, passage, num_last))

print("doing sets")
sets = []
for query, ents, _, _ in result:
    if query in queries:
        sets.append({queries[query]: None})
    else:
        sets.append({k: None for k in lens[len(ents)]})
sets[0] = {0: None}
for i, (past, cur) in enumerate(zip(sets[:-1], sets[1:])):
    mn = min(past.keys())
    td = []
    for c in cur:
        if c < mn+1:
            td.append(c)
    for j in td:
        del sets[i+1][j]


print(sets[1])
print(sets[2])
print(sets[3])
exit()

result = result[1:]
print(result[:10])
print(res[:10])
exit()
source, labels = backup.iloc[:, 0], backup.iloc[:, 1]
print(source)
