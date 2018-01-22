import csv
import jieba
import jieba.posseg as pseg

read = csv.DictReader(open('baike.csv','r'))

illness = csv.writer(open('illness.csv', 'w', newline='', encoding='utf8'))

symptom = csv.writer(open('symptom.csv', 'w', newline='', encoding='utf8'))

for r in read:

    # 病情语料库
    # illness.writerow([r['title'] + ' 20 '+ 'mi'])
    #
    # for s in r['typical_symptom'].split(',')[:-1]:
    #     symptom.writerow([s + ' 20 ' + 'ms'])
    # print()
    result = pseg.cut(r['introduction'])

    for w in result:
        print(w.word, "/", w.flag, ", ", end=' ')
        print()

