import requests
import html
import json
from stem import Signal
from stem.control import Controller
import time
import shutil


def translate(lines, translations=None, char_limit=50_000, tor_port=9050, cnt_port=9051, cb=lambda x: None):
    if translations is None:
        translations = {}
    translations[''] = ''
    total = len(lines)
    session = requests.session()
    if tor_port is not None:
        if cnt_port is not None:
            controller = Controller.from_port(port=cnt_port)
            controller.authenticate()
            controller.signal(Signal.NEWNYM)
        session.proxies = {'http': f'socks5h://localhost:{tor_port}', 'https': f'socks5h://localhost:{tor_port}'}
    while True:
        lines = [x.strip() for x in lines]
        lines = [x for x in lines if x not in translations]
        if not lines:
            return translations
        to_translate = []
        chars = 0
        for line in lines:
            chars += len(line)
            if chars > char_limit:
                break
            to_translate.append(line)
        text = '\n'.join(to_translate)
        # print(to_translate)
        translate_url = "https://translate.googleusercontent.com/translate_f"
        r = session.post(translate_url,
                         data=dict(sl='ru', tl='en', hl='en-US', ie='UTF-8', js='y', prev='_t', ),
                         files=dict(file=('source.txt', text, 'text/plain')))
        txt = r.text
        if "Your client does not have permission to get URL" in txt or \
           "is inappropriate for the URL" in txt or \
           "Our systems have detected unusual traffic from your computer network" in txt:
            print("Restart")
            if cnt_port is not None:
                controller.authenticate()
                controller.signal(Signal.NEWNYM)
            time.sleep(10)
            continue
        txt = html.unescape(txt[5:-6]).split('\n')
        update_dict = {k: v for k, v in zip(to_translate, txt) if not (russian(v))}
        translations.update(update_dict)
        print("Left:", f'{len(lines)}/{total}', "Efficiency:", (len(update_dict) + 1) / (len(to_translate) + 1) * 100)
        cb(translations)
        time.sleep(3)


abcd = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
def russian(x):
    x = x.lower()
    if len([y for y in x if y in abcd]) / max(len(x), 1) > 0.03:
        return True
    return False


def translate_all(source, save):
    try:
        translations = json.load(open(save))
    except FileNotFoundError:
        print(" Not found")
        translations = {}
    lines = [x.strip() for x in source]
    def cb(t):
        try:
            shutil.move(save, "translations/translation-bk.json")
            json.dump(t, open(save, 'w'), ensure_ascii=False)
        except KeyboardInterrupt:
            cb(t)
    translations = translate(lines, translations=translations, cb=cb)
    cb(translations)
    # json.dump(translations, open(save, 'w'))
    return translations


if __name__ == '__main__':
    file = "train"
    save = f"translations/{file}.json"
    source = open(f"translations/{file}.ru")
    translate_all(source, save)
