# -*- coding: utf-8 -*-
# @Author: huzhu
# @Date:   2019-10-22 20:37:58
# @Last Modified by:   huzhu
# @Last Modified time: 2019-10-23 19:40:25

from scipy.spatial import KDTree
import codecs
import time
import random
import math
import numpy as np
xxxx=[]
# import plot_clusters as plot

def DBSCAN(X, epsilon, MinPts):
    C = 0
    n = len(X)
    IDX = np.zeros(n)
    visited = np.zeros(n)
    # t=time.time_ns()
    kd = KDTree(X)
    # print((time.time_ns()-t)/1e6)
    # t=time.time_ns()
    for i in range(0, n):
        if visited[i] == 0:
            visited[i] = 1
            t=time.time_ns()
            N = kd.query_ball_point(X[i], epsilon)[0]
            xxxx.append((time.time_ns() - t) / 1e6)
            # print()
            if len(N) >= MinPts:
                C = C+1
                IDX, visited = ExpandCluster(i,X, C, N, kd, epsilon, MinPts, IDX, visited)
    # print((time.time_ns() - t) / 1e6)
    return IDX


def ExpandCluster(i,X, C, N, kd, epsilon, MinPts, IDX, visited):
    IDX[i] = C
    
    k = 0
    while True:
        j = N[k]
        if visited[j] == 0:
            visited[j] = 1
            t=time.time_ns()

            N2 = kd.query_ball_point(X[j], epsilon)[0]
            xxxx.append((time.time_ns() - t) / 1e6)

            if len(N2) >= MinPts:
                N.extend(N2)
        
        if IDX[j] == 0:
            IDX[j] = C
        
        k = k + 1
        if k >= len(N):
            break

    return IDX,visited

if __name__ == '__main__':
    data = list()
    with codecs.open("five_cluster.txt", encoding='utf-8') as f:
        for line in f.readlines():
            item = [float(x) for x in line.split(" ")]
            data.append(item)
    data = np.array(data)
    data= np.mat(data)
    t = time.time_ns()
    IDX = DBSCAN(data[:, 1:3], 0.4, 9)
    print((time.time_ns()-t)/1e6)
    # plot.plot_clusters(data[:, 1:3],IDX,"DBSCAN")
    print(np.average(xxxx))
