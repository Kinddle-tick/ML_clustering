import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import Birch

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

DataMat=loadDataSet('five_cluster.txt')
cluster_num=5
X=DataMat[:,[1,2]]
g_truth=DataMat[:,0]
# for 'five_cluser.txt':threshold=1.5,branching_factor=20
# for 'spiral.txt':不适用
# for 'ThreeCircles.txt':不适用
# for 'Twomoons.txt':不适用
y_pred = Birch(n_clusters = cluster_num,threshold=1.5,branching_factor=20).fit_predict(X)

plt.subplot(211)
plt.suptitle('Clustering by BIRCH',fontsize=8)
plt.scatter(X[:, 0], X[:, 1],s=2, c=np.transpose(g_truth))
plt.subplot(212)
plt.scatter(X[:, 0], X[:, 1],s=2, c=y_pred)
plt.show()
#print ("Calinski-Harabasz Score"), metrics.calinski_harabaz_score(X, y_pred)