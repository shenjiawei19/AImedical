import jieba
import jieba.posseg as pseg

jieba.load_userdict("illness.csv")
jieba.load_userdict("symptom.csv")


test_sent = '请问被毛竹不小心砍伤了, 晚上有点头疼脑热, 是不是中毒了, 还打喷嚏,该如何治疗'
result = pseg.cut(test_sent)

for w in result:
    print(w.word, "/", w.flag, ", ", end=' ')