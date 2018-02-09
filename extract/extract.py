# -*- coding: utf-8 -*-
# author 沈佳巍（蜗牛）
import csv
import jieba.posseg as jpsg


def sy_il():
    rf = csv.DictReader(open('./source/baike.csv', 'r', encoding='gbk'))
    msy = csv.writer(open('./target/medical_symptom.csv', 'w+', encoding='utf8', newline=''))
    mil = csv.writer(open('./target/medical_illness.csv', 'w+', encoding='utf8', newline=''))

    sy = []
    for r in rf:
        mil.writerow([r['title']+' '+'100'+' '+'mil'])
        for i in r['typical_symptom'].split(',')[:-1]:
            if i not in sy:
                sy.append(i)
    for s in sy:
        msy.writerow([s + ' ' + '100' + ' ' + 'msy'])


def nvo():
    rf = csv.DictReader(open('./source/baike.csv', 'r', encoding='gbk'))
    mnf = csv.writer(open('./target/medicaln.csv', 'w+', encoding='utf8', newline=''))
    mvf = csv.writer(open('./target/medicalv.csv', 'w+', encoding='utf8', newline=''))
    mof = csv.writer(open('./target/medicalo.csv', 'w+', encoding='utf8', newline=''))
    mn = []
    mv = []
    mo = []
    for r in rf:
        for word, tag in jpsg.cut(r['introduction']):
            if tag not in ['x', 'eng']:
                if tag == 'n':
                    new_word = word + ' ' + '80' + ' m' + tag
                    if new_word not in mn:
                        mn.append(new_word)
                elif tag == 'v':
                    if len(word) > 1:
                        new_word = word + ' ' + '80' + ' m' + tag
                        if new_word not in mv:
                            mv.append(new_word)
                elif tag in ['l', 'vn', 'a', 'd', 'c', 'i',
                             'ad', 'b', 'nz', 'nt', 'ns', 'nr',
                             'an', 'zg', 't']:
                    new_word = word + ' ' + '50' + ' m' + tag
                    if new_word not in mo:
                        mo.append(new_word)
                else:
                    pass
    for m in mo:
        mof.writerow([m])
    for m in mn:
        mnf.writerow([m])
    for m in mv:
        mvf.writerow([m])


def c_open(file):
    with open(file, mode='rb') as f:
        for _ in f:
            try:
                yield _.decode('gbk')
            except:
                pass


def training():
    a = csv.DictReader(c_open('source/data.csv'))
    w = csv.DictWriter(open('training/training.csv', mode='w', newline='', encoding='utf8'), fieldnames=['qus', 'clf', 'ans'])
    w.writeheader()
    for i in a:
        row = {
            'qus': (i['BT00'],i['WT00']),
            'clf': (i['YJFL'],i['EJFL'],i['SJFL']),
            'ans': (i['HDNN'])
        }
        w.writerow(row)

if __name__ == '__main__':
    sy_il()                 # 病情症状的专项分词
    nvo()                   # 名词动词及其他词性的专项分词
    training()