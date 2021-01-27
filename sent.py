from cache import global_cache, mem
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import stopwords
import fasttext
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


@mem.cache
def embed_sentence(sent, model="paraphrase-xlm-r-multilingual-v1"):
    bert = global_cache("sentence", lambda: SentenceTransformer(model))
    return bert.encode(sent)


def sent_sim(emb1, emb2):
    return util.pytorch_cos_sim(emb1, emb2)[0].item()


allowed_russian = set("йцукенгшщзхъфывапролджэячсмитьбю1234567890qwertyuiopasdfghjklzxcvbnm")


@mem.cache
def word2vec(sent, ru=False):
    if ru:
        model = global_cache("word2vec_ru", lambda: fasttext.load_model("models/fasttext.ru.bin"))
        words_ = []
    else:
        model = global_cache("word2vec", lambda: fasttext.load_model("models/fasttext.bin"))
        words_ = global_cache("stopwords", lambda: stopwords.words())
    words = [''.join(char.lower() for char in word if char in allowed_russian) for word in nltk.tokenize.word_tokenize(sent)]
    embs = [model[word] for word in words if word and word not in words_]
    return embs


def word_sim(sent1, sent2):
    sent1 = np.asarray(sent1)
    sent2 = np.asarray(sent2)
    if sent2.shape == (0,):
        return None
    if sent1.shape == (0,):
        return None
    return cosine_similarity(sent1, sent2)


if __name__ == '__main__':
    print(word2vec("Hello world. This is a test."))
