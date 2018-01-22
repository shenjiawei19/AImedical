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
'''

class sen(object):
    """
    pass
    """
    def __init__(self, cus_files=None):

        if type(cus_files) is list:
            for cf in cus_files:
                for s in open(cf,'r'):
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
    for sen in [sen1,sen2,sen3,sen4]:
        print(sen)
        for x, w in a.extract_tags(sen,topK=10, withWeight=True, allowPOS=('msy','mil','mn','mv','ml', 'mvn', 'ma', 'md', 'mc', 'mi',
                             'mad', 'mb', 'mnz', 'mnt', 'mns', 'mnr','man', 'mzg', 'mt'),withFlag=True):
            print('%s %s %s' % (x.word,x.flag, w))
        print()
    # jieba.analyse.TextRank()
    # for x, w in jieba.analyse.textrank(s, topK=20, withWeight=True, allowPOS=('md','n','nz')):
    #     print('%s %s' % (x, w))
    exit()
    print('=' * 40)
    print('4. 词性标注')
    print('-' * 40)
    # cus_dict_file = 'data/cus_demo.txt'
    # jieba.load_userdict(cus_dict_file)
    # string1 = "国内掀起了大数据、云计算的热潮。"
    # seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
    # print("Full Mode: " + "/ ".join(seg_list))  # 全模式
    #
    # seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
    # print("Default Mode: " + "/ ".join(seg_list))  # 精确模式
    # # jieba.add_word('杭研大厦')
    # seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
    # print(", ".join(seg_list))
    #
    # seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
    # print(", ".join(seg_list))
    #
    # print('/'.join(jieba.cut('「台中」正确应该不会被切开', HMM=False)))
    # jieba.suggest_freq('台中', True)
    #
    # print('/'.join(jieba.cut('「台中」正确应该不会被切开', HMM=False)))
    # seg_list = sentens.cut("腰椎间盘突出有什么办法可以得到很好的治疗？",cut_all=True)
    # print("Full Mode: " + "/ ".join(seg_list))
    # jieba.add_word('大数据')
    # sentens.add_words(new_words=['大数据','欧亚置业'])
    # seg_list = ("国内掀起了大数据、云计算的热潮。",)


    # print(",".join(tags))

    # seg_list = jbpos.cut("国内掀起了大数据、云计算的热潮。",)
    # for word, flag in seg_list:
    #     print('%s %s' % (word, flag))
    # print("Full Mode: " + "/ ".join(seg_list))
    # exit()
    # s = "此外在大数据领域，公司拟对全资子公司吉林欧亚置业有限公司增资4.3亿元，" \
    #     "增资后，吉林欧亚置业注册资本由7000万元增加到5亿元。吉林欧亚置业主要经营范围为房地产开发及百货零售等业务。" \
    #     "目前在建吉林欧亚城市商业综合体项目。2013年，实现营业收入0万元，实现净利润-139.13万元。"
    # seg_list = jbpos.cut(sen, HMM=False)
    # del_word('怎么办')
    # for word, flag in seg_list:
    #     print('%s %s' % (word, flag))
    # exit()
    # print(jieba.analyse.extract_tags(sen))
    #