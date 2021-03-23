from scipy.spatial import KDTree
import codecs
import time
import random
import math
import numpy as np

def DBSCAN(X, epsilon, MinPts):
    C = 0
    n = len(X)
    IDX = np.zeros(n)
    visited = np.zeros(n)
    kd = KDTree(X)

    for i in range(0, n):
        if visited[i] == 0:
            visited[i] = 1
            N = kd.query_ball_point(X[i], epsilon)[0]

            if len(N) >= MinPts:
                C = C+1
                IDX, visited = ExpandCluster(i,X, C, N, kd, epsilon, MinPts, IDX, visited)
    
    return IDX


def ExpandCluster(i,X, C, N, kd, epsilon, MinPts, IDX, visited):
    IDX[i] = C
    
    k = 0
    while True:
        j = N[k]
        if visited[j] == 0:
            visited[j] = 1
            N2 = kd.query_ball_point(X[j], epsilon)[0]

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
    with codecs.open("D:\\大三下\\人工智能基础\\Thecode\\Chapter2\\data\\five_cluster.txt", encoding='utf-8') as f:
        for line in f.readlines():
            item = [float(x) for x in line.split(" ")]
            data.append(item)
    data = np.array(data)
    IDX = DBSCAN(data[:, 1:3], 0.4, 9)
    # plot.plot_clusters(data[:, 1:3],IDX,"DBSCAN")
