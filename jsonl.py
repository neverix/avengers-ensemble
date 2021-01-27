import json


def load_jsonl(fpath):
    with open(fpath) as jsonl:
        for line in jsonl:
            if line:
                yield json.loads(line)


def write_jsonl(fpath, jsonl):
    open(fpath, 'w').write('\n'.join([json.dumps(x) for x in jsonl]))


