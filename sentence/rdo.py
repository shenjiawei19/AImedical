import csv
from sentence.sensplit import sen, Mytfidf
import os
new_dict = ['../data/' + f for f in os.listdir('../data') if f.endswith('csv')]
sentens = sen(cus_files=new_dict)

a = Mytfidf()



cr = csv.DictReader(open('../30W.csv', 'r', encoding='gbk'))
n = 0
for c in cr:
    try:
        sen = c['WT00']
        print(sen)
        for x, w in a.extract_tags(sen, topK=10, withWeight=True,
                                   allowPOS=('msy', 'mil', 'mn', 'mv', 'ml', 'mvn', 'ma', 'md', 'mc', 'mi',
                                             'mad', 'mb', 'mnz', 'mnt', 'mns', 'mnr', 'man', 'mzg', 'mt'),
                                   withFlag=True):
            print('%s %s %s' % (x.word, x.flag, w))
        # print(c['HDNN'])
        print()
    except Exception as e:
        pass
    n+=1
    if n == 5:
        exit()