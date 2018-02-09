import csv
from ast import literal_eval


def c_open(file):
    with open(file, mode='rb') as f:
        for _ in f:
            try:
                yield _.decode('gbk')
            except:
                pass

train = csv.DictReader(open('./training/training.csv',encoding='utf8'))


for t in train:
    print(eval(t['qus']),t['clf'],t['ans'])
    print(type(literal_eval(t['qus'])), type(t['clf']), t['ans'])
    # print(t.split(','))
    exit()
