import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import Birch
from collections import Counter
import time

''' 将.txt文件导入到array中'''
def loadDataSet(filename):
    data=[]
    fr=open(filename)
    for line in fr.readlines():
        curLine=line.strip('\n').split(' ')
        fltLine=list(map(float,curLine))
        data.append(fltLine)
    dataMat= np.array(data)
    return dataMat

'''计算准确率'''
def Coopcheckdiv(x,y):
    '''
    :param x: 一行分类数据
    :param y: 一行分类数据,与x等长
    :return: 两份数据的"乐观准确率"
    '''
    prime_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101,
                  103,107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211,
                  223,227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293]
    divx=set(x)
    xdivnum=len(divx)
    divy=set(y)
    ydivnum=len(divy)
    num=2
    # 防止串号
    while prime_list[num]<len(divx)+len(divy):
        num+=1
    for i in divx:
        if i<0:  # 噪声不属于任何一类
            x[np.where(x == i)] = 0
            xdivnum-=1
            continue
        x[np.where(x == i)] = prime_list[num]
        num+=1
    for i in divy:
        if i<0:
            x[np.where(x == i)] = 0
            ydivnum-=1
            continue
        y[np.where(y == i)] = prime_list[num]
        num += 1
    rightlist = Counter((x*y)[x*y!=0]).most_common(min(xdivnum,ydivnum))
    right = sum(np.array(rightlist)[:,1])
    all =len(x)
    return right/all

DataMat=loadDataSet('five_cluster.txt')
cluster_num=5
X=DataMat[:,[1,2]]
g_truth=DataMat[:,0]
# for 'five_cluser.txt':threshold=1.5,branching_factor=20
# for 'spiral.txt':不适用
# for 'ThreeCircles.txt':不适用
# for 'Twomoons.txt':不适用
t0=time.time()
y_pred = Birch(n_clusters = cluster_num,threshold=1.5,branching_factor=20).fit_predict(X)
t1=time.time()

plt.subplot(211)
plt.suptitle('Clustering by BIRCH',fontsize=16)
plt.scatter(X[:, 0], X[:, 1],s=2, c=np.transpose(g_truth))
plt.subplot(212)
plt.scatter(X[:, 0], X[:, 1],s=2, c=y_pred)
plt.show()

'''
result_accuracy=Coopcheckdiv(a,b)
print('Accuracy Rate Is:')
print(result_accuracy)
'''
print('Processing Time Is:')
print(t1-t0)