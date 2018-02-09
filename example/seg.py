import analyse as aly
import os
import csv
import ast


def test():
    new_dict = ['../extract/target/' + f for f in os.listdir('../extract/target/') if f.endswith('csv')]
    aly.sen(cus_files=new_dict)

    train = csv.DictReader(open('../extract/training/training.csv',encoding='utf8'))
    an = aly.Mytfidf()

    stop = ["时"]
    n = 0
    for i in train:
        n += 1
        for qus in tuple(ast.literal_eval(i['qus'])):
            print(qus)
            for x, w in an.extract_tags(qus, topK=10, withWeight=True,
                                        allowPOS=('msy', 'mil', 'mn', 'mv', 'ml', 'mvn', 'ma', 'md', 'mc', 'mi',
                                                  'mad', 'mb', 'mnz', 'mnt', 'mns', 'mnr', 'man', 'mzg', 'mt'),
                                        withFlag=True):
                print('%s %s %s' % (x.word, x.flag, w))
                qus = qus.replace(x.word, "")
            for x, w in an.extract_tags(qus, topK=10, withWeight=True, withFlag=True, HMM=True):
                if x not in stop:
                    print(x,w)
            print("###########################")
        if n == 1:
            exit()

if __name__ == '__main__':
    # test()
    print(max(((123124,8),(123123,9))))
    # print(16.2 / 2)
    # a = "务农砍毛竹时手指被砸伤"
    # a.replace()
    # new_dict = ['../extract/target/' + f for f in os.listdir('../extract/target/') if f.endswith('csv')]
    # print(new_dict)
    # sentence = aly.sen(cus_files=new_dict)
    # sen1 = "如何提高免疫力？"
    # sen2 = "腰椎间盘突出有什么办法可以得到很好的治疗？"
    # sen3 = "胃疼应该怎么办"
    # sen4 = "前列腺增生该如何治疗"
    # an = aly.Mytfidf()
    #
    # for sen in [sen1, sen2, sen3, sen4]:
    #     print(sen)
    #     for x, w in an.extract_tags(sen, topK=10, withWeight=True,
    #                                allowPOS=('msy', 'mil', 'mn', 'mv', 'ml', 'mvn', 'ma', 'md', 'mc', 'mi',
    #                                          'mad', 'mb', 'mnz', 'mnt', 'mns', 'mnr', 'man', 'mzg', 'mt'), withFlag=True):
    #         print('%s %s %s' % (x.word, x.flag, w))
    #     print()
