import numpy as np
import matplotlib.pyplot as plt
# 选择聚类方法：clique 类
from pyclustering.cluster.clique import clique
# clique 可视化
from pyclustering.cluster.clique import clique_visualizer
import time
from collections import Counter
from Myfunc import Coopcheckdiv


def loadDataSet(filename):
    data=[]
    fr=open(filename)
    for line in fr.readlines():
        curLine=line.strip('\n').split(' ')
        fltLine=list(map(float,curLine))
        data.append(fltLine)
    data_M = np.array(data)
    return data_M

    # file_list = ["five_cluster.txt", "spiral.txt",
    #              "ThreeCircles.txt", "Twomoons.txt"]
data_M = loadDataSet('Twomoons.txt')
TestData = data_M[:,[1,2]]
g_truth = data_M[:,0]
# 创建 CLIQUE 算法进行处理
# 定义每个维度中网格单元的数量
intervals = 15
# 密度阈值
threshold = 10
clique_instance = clique(TestData, intervals, threshold)

'''
five_cluster.txt  intervals = 15,threshold = 10  0.812    t=0.039999961853027344
spiral.txt no
ThreeCircles.txt  intervals = 20,threshold = 1  0.990563419372745   t=0.0800008773803711
Twomoons.txt  intervals = 15,threshold = 1    0.9880159786950732   t=0.04000043869018555
'''


# 开始聚类过程并获得结果
t1 = time.time_ns()
clique_instance.process()
t2 = time.time_ns()
clique_cluster = clique_instance.get_clusters()  # allocated clusters
# 被认为是异常值的点（噪点）
noise = clique_instance.get_noise()
# CLIQUE形成的网格单元
cells = clique_instance.get_cells()

#print(clique_cluster[3])

zeros = [0]*len(g_truth)

for each in clique_cluster[0]:
    zeros[each] = 3
for each in clique_cluster[1]:
    zeros[each] = 2
for each in clique_cluster[2]:
    zeros[each] = 4
for each in clique_cluster[3]:
    zeros[each] = 5
# for each in clique_cluster[4]:
#     zeros[each] = 1
g_answer = zeros

#print(g_answer)
sum = 0
for i in range(len(g_truth)):
    if g_answer[i] == g_truth[i]:
        sum+=1

rate = sum/len(g_truth)

print(rate)

#print(g_truth)
#print(g_answer)

print("Amount of clusters:", len(clique_cluster))


print(clique_cluster)
print(f"运行时间：{(t2-t1)/1e6}ms")   #运行时间
# 显示由算法形成的网格
clique_visualizer.show_grid(cells, TestData)
# 显示聚类结果
clique_visualizer.show_clusters(TestData, clique_cluster, noise)  # show clustering results

#  准确率
print(f"准确率：{Coopcheckdiv(np.array(g_answer),data_M[:,0])}")
