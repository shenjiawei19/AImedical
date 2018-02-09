import csv
import jieba.posseg as jpsg


rf = csv.DictReader(open('baike.csv', 'r', encoding='gbk'))
msy = csv.writer(open('medical_symptom.csv', 'w+', encoding='utf8'))
# mil = csv.writer(open('medical_illness.csv', 'w+', encoding='utf8'))

sy = []
for r in rf:
    # mil.writerow([r['title']+' '+'20'+' '+'mil',])
    for i in r['typical_symptom'].split(',')[:-1]:
        if i not in sy:
            sy.append(i)
for s in sy:
    msy.writerow([s + ' ' + '20' + ' ' + 'msy'])


def nvo():
    rf = csv.DictReader(open('baike.csv', 'r', encoding='gbk'))
    mnf = csv.writer(open('medicaln.csv', 'w+', encoding='utf8'))
    mvf = csv.writer(open('medicalv.csv', 'w+', encoding='utf8'))
    mof = csv.writer(open('medicalo.csv', 'w+', encoding='utf8'))
    # mil = csv.writer(open('medical_illness.csv', 'w+', encoding='utf8'))
    mn = []
    mv = []
    mo = []
    for r in rf:
        for word , tag in jpsg.cut(r['introduction']):
            if tag not in ['x', 'eng']:
                if tag == 'n':
                    new_word = word + ' ' + '19' + ' m' + tag
                    if new_word not in mn:
                        mn.append(new_word)
                    # pass
                elif tag == 'v':
                    # pass
                    if len(word) > 1:
                        new_word = word + ' ' + '19' + ' m' + tag
                        if new_word not in mv:
                            mv.append(new_word)
                elif tag in ['l', 'vn', 'a', 'd', 'c', 'i',
                             'ad', 'b', 'nz', 'nt', 'ns', 'nr',
                             'an', 'zg', 't']:
                    new_word = word + ' ' + '19' + ' m' + tag
                    if new_word not in mo:
                        mo.append(new_word)
                    # pass
                    # print(len(word),tag)
                # elif tag == 'v':
                #     print(word,tag)
                # elif tag == 'v':
                #     print(word,tag)
                else:
                    # print(word, tag)
                    pass
    for m in mo:
        mof.writerow([m])
    for m in mn:
        mnf.writerow([m])
    for m in mv:
        mvf.writerow([m])
        # exit()
# nvo()