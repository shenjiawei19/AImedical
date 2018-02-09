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


class sen(object):
    """
    pass
    """
    def __init__(self, cus_files=None):

        if type(cus_files) is list:
            for cf in cus_files:
                for s in open(cf, 'r', encoding='utf8'):
                    del_word(s.split()[0])
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
    del_word =  del_word

class Mytfidf(jbal.TFIDF):

    def __init__(self):
        super(Mytfidf, self).__init__()

    def extract_tags(self, sentence, topK=20, withWeight=False, allowPOS=(), withFlag=False, HMM=False):
        if allowPOS:
            allowPOS = frozenset(allowPOS)
            words = self.postokenizer.cut(sentence, HMM=HMM)
        else:
            words = self.tokenizer.cut(sentence,HMM=HMM)
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

