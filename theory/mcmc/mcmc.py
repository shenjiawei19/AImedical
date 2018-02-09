'''
Date:20180111
@author: 蜗牛
'''

import matplotlib.pyplot as plt
import numpy as np
import random
from pylab import mpl
from math import pi
mpl.rcParams['font.sans-serif'] = ['SimHei']

##########################
# title = "蒙特卡罗算法圆周率计算"
# fig = plt.figure(title)
# ax = fig.add_subplot(111)
#
# height = [random.random() for _ in range(2000)]
# weight = [random.random() for __ in range(2000)]
# tmp = [(x,y) for x, y in  zip(height, weight) if (x-0.5)**2 + (y-0.5)**2 <= 0.25]
# height2, weight2 = zip(*tmp)
#
# theta = np.arange(0, 2 * np.pi ,2 * np.pi / 1000)
# x = np.cos(theta)/2 + 0.5
# y = np.sin(theta)/2 + 0.5
#
# plt.scatter(height,weight,c='red',s=10,alpha=1,marker='o')
# plt.scatter(height2,weight2,c='blue',s=10,alpha=1,marker='o')
# plt.scatter(x,y,c='blue',s=5,alpha=0.4,marker='o')
#
# ax.set_xlabel("横坐标")
# ax.set_ylabel("纵坐标")
# ax.set_title('面积')
#
# print(len(height2)/len(height)*4)
# print(pi/(len(height2)/len(height)*4)-1)
# plt.show()
#####################################
# title = "蒙特卡罗算法积分计算"
# fig = plt.figure(title)
# ax = fig.add_subplot(111)
# x, y, z = zip(*[(i/2000 , (i/2000)**2, random.uniform(0,1)) for i in range(2000)])
# labels = ["y = x**2"]
# plt.stackplot(x, y, labels=labels)
# plt.scatter(x, z, c='red',s=10,alpha=1,marker='o')
# pa = [y for y,z in zip(y, z) if z < y]
# print(len(pa)/2000)
# ax.set_xlabel("横坐标")
# ax.set_ylabel("纵坐标")
# ax.set_title('积分计算')
# plt.show()

######################################
# title = "蒙特卡罗算法高级场景交通领域的应用"
# 当前速度是 v
# 如果前面没车，(距离为d，且 d >= v), 它在下一秒的速度会提高到 v + 1 ，直到达到规定的最高限速
# 如果前面有车，(距离为d，且 d < v)，那么它在下一秒的速度会降低到 d - 1
# 此外，司机还会以概率 p 随机减速， 将下一秒的速度降低到 v - 1
#
# 全路场 S = 100000m
# 模拟 1000 量 车 v = （40,60)m/s 每辆车相距 200米

# num = [i for i in range(0, 100)]
# x = [i for i in range(0, 2000,20)]
# t = [i for i in range(60)]
# v = [random.uniform(4,6) for _ in range(100)]
#
# x = [[num, x+v, v+1] for num, x, v in zip(num, x, v)]
# X = []
# Y = []
# for i in t:
#     for n, s, v in x:
#         if n < 99:
#             d = abs(x[n+1][1] - x[n][1])
#         else:
#             d = 1000
#         if d >= v:
#             x[n][1] = s + v
#             x[n][2] = v + 1
#             if random.randint(1, 3) == 2:
#                 x[n][2] = v - 2 if v - 2 > 0 else 1
#         else:
#             x[n][1] = s + v
#             x[n][2] = d -1
#
#     X.extend([y[1] for y in x])
#     Y = Y+[i]*100
#
# fig = plt.figure(title)
# ax = fig.add_subplot(111)
# plt.scatter(Y,X,c='black',s=0.1,alpha=1,marker='o')
#
# ax.set_xlabel("时间")
# ax.set_ylabel("距离")
# ax.set_title('交通阻塞图')
#
# plt.show()

##########################
# title = "蒙特卡罗算法圆周率计算"
# fig = plt.figure(title)
# ax = fig.add_subplot(111)
#
#
# def mc(_):
#     height = [random.random() for _ in range(2000)]
#     weight = [random.random() for __ in range(2000)]
#     tmp = [(x,y) for x, y in  zip(height, weight) if (x-0.5)**2 + (y-0.5)**2 <= 0.25]
#     height2, weight2 = zip(*tmp)
#     return len(height2)/len(height)*4
#
# num = 100
# value = [mc(_) for _ in range(num)]
# mean_v = np.mean(value)
# min_v = np.min(value)
# max_v = np.max(value)
# std_v = np.std(value)
# var_v = np.var(value)
# print(num)
# print(mean_v)
# print(min_v)
# print(max_v)
# print(std_v)
# print(var_v)
# plt.plot(range(num), value, 'g-')
# plt.show()

# 采样数, 期望值, 最小值, 最大值, 标准差, 方差
# 100 3.14348 3.018 3.236 0.0391485580833 0.0015326096/100
# 300 3.1403 3.038 3.24 0.0357540207529 0.00127835/300
# 500 3.144192 3.044 3.244 0.0350975089714 0.001231835136/500
# 1000 3.141402 3.002 3.238 0.0356908727268 0.001273838396/1000
# 10000 3.1415892 3.008 3.266 0.0373316493523 0.00139365204336/10000

################################
title = "蒙特卡罗算法积分计算的重要新采样"
fig = plt.figure(title)
ax = fig.add_subplot(111)

# 样本数据总和 N
N = 20000

# 概率分布随机次数
p = 10000

x, y, z = zip(*[(i/p , (i/p)**2, random.uniform(0,7)) for i in range(N)])
labels = ["y = x**2"]

plt.stackplot(x, y, labels=labels)
plt.scatter(x, z, c='red',s=1,alpha=0.4,marker='o')

pa = [y for y,z in zip(y, z) if z < y]
print('总体采样-分布图结果', len(pa)/N)

def ra():
    n = random.randint(0,N-1)
    return (x[n], y[n], z[n])

pca = []
# 重要性采样次数 n
Ni = 10000
# 每次抽取个数
Z = 100
# 期望分布范围
ma , mi = 0.4 ,0.31
n = 0
for t in range(Ni):
    pre = [ra() for i in range(Z)]
    pa = [(x, y, z) for x, y, z in pre if z < y]
    rag = len(pa)/Z
    if  (ma > rag) and  (rag > mi):
        pca += pre
        print(len(pa)/Z)
        n+=1
    if n == 100:
        break
x, y, z = zip(*pca)
plt.scatter(x, z, c='black',s=10,alpha=1,marker='o')


pa = [y for y,z in zip(y, z) if z < y]
print('重要性-分布图结果', len(pa)/(100*n))
var_v = np.var(pa)
print("重要采样的方差", var_v)
ax.set_xlabel("横坐标")
ax.set_ylabel("纵坐标")
ax.set_title('分布图')
plt.show()
