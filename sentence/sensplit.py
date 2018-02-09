# -*- coding: utf-8 -*-
# author 沈佳巍（蜗牛）
import jieba
import jieba.posseg as jbpos
import jieba.analyse as jbal
from jieba import load_userdict, cut, \
    add_word, suggest_freq, cut_for_search, \
    lcut, lcut_for_search, del_word
from operator import itemgetter
import os

'''
<<<<<<< HEAD
分词demo
=======
词性说明:
a:形容词
d:副词
i:成语
m:数词
n:名词
nr:人名
ns:地名
nt:机构团体
nz:其他专有名词
t:时间
v:动词
x:标点符号
f:方位词
un:未知
>>>>>>> 0d1c900770e512cd9233fbebb364a5f1a819f5a8
'''

class sen(object):
    """
    pass
    """
    def __init__(self, cus_files=None):

        if type(cus_files) is list:
            for cf in cus_files:
                for s in open(cf, 'rb'):
                    del_word(s.split()[0].decode())
                load_userdict(cf)
        assert type(cus_files) is list, "cus_files must be a files list"

    @staticmethod
    def add_words(new_words=None):
        if type(new_words) is list:
            for nw in new_words:
                add_word(nw,)
        assert type(new_words) is list, "new_words must be a sentense list"

    @staticmethod
    def merge_words(merge_words=None):
        if type(merge_words) is list:
            for mw in merge_words:
                suggest_freq(mw)
        assert type(merge_words) is list, "new_words must be a sentense list"


    cut = cut
    search = cut_for_search
    lcut = lcut
    lsearch = lcut_for_search

class Mytfidf(jbal.TFIDF):

    def __init__(self):
        super(Mytfidf, self).__init__()

    def extract_tags(self, sentence, topK=20, withWeight=False, allowPOS=(), withFlag=False):
        if allowPOS:
            allowPOS = frozenset(allowPOS)
            words = self.postokenizer.cut(sentence, HMM=False)
        else:
            words = self.tokenizer.cut(sentence,HMM=False)
        freq = {}
        for w in words:
            if allowPOS:
                if w.flag not in allowPOS:
                    continue
                elif not withFlag:
                    w = w.word
            wc = w.word if allowPOS and withFlag else w
            freq[w] = freq.get(w, 0.0) + 1.0
        total = sum(freq.values())
        for k in freq:
            kw = k.word if allowPOS and withFlag else k
            freq[k] *= self.idf_freq.get(kw, self.median_idf) / total

        if withWeight:
            tags = sorted(freq.items(), key=itemgetter(1), reverse=True)
        else:
            tags = sorted(freq, key=freq.__getitem__, reverse=True)
        if topK:
            return tags[:topK]
        else:
            return tags

if __name__ == '__main__':
    new_dict = ['data/'+f for f in os.listdir('data') if f.endswith('csv')]
    sentens = sen(cus_files=new_dict)
    sen1 = "如何提高免疫力？"
    sen2 = "腰椎间盘突出有什么办法可以得到很好的治疗？"
    sen3 = "胃疼该怎么办"
    sen4 = "前列腺增生该如何治疗"
    a = Mytfidf()
