from diskcache import Cache


class Memory(Cache):
    def __init__(self, *args, **kwargs):
        super(Memory, self).__init__(*args, **kwargs)
        self.reset('cull_limit', 0)
        self.reset('size_limit', int(50e6))

    def cache(self, func):
        return super(Memory, self).memoize()(func)


_cache_dict = {}
mem2 = Memory("cache_dir")
import joblib
mem = joblib.Memory("cache_dir", verbose=False)


def global_cache(word, init):
    if word not in _cache_dict:
        _cache_dict[word] = init()
    return _cache_dict[word]
