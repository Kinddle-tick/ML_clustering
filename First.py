import pyqtgraph as pg
import pandas as pd
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
from Myfunc import DistanceMatrix
# def DistanceMatrix(point:[[2,11],[3,5],]):
#     G=np.dot(point,point.values.T)
#     H=np.tile(np.diag(G),(len(G),1))
#     D=H+H.T-2*G
#     return D

def Prim(D_M):
    D_df = pd.DataFrame(D_M)
    lens = len(D_M)
    V= {0}
    ALL = set([i for i in range(lens)])
    E=[]
    while len(V)<lens:
        tmp= D_df.iloc[list(V),list(ALL-V)]
        x,y = np.where(D_M == np.min(np.array(tmp)))
        line = [x[0],y[0]]
        E.append(line)
        V=V|set(line)
    return E

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

w = pg.GraphicsLayoutWidget(show=True)
w.setWindowTitle('pyqtgraph example: GraphItem')
v = w.addViewBox()
v.setAspectLocked()

g = pg.GraphItem()
v.addItem(g)

# file_list = ["five_cluster.txt", "spiral.txt", "ThreeCircles.txt", "Twomoons.txt"]
file_list = ["Twomoons.txt"]

for file in file_list:
    train = pd.read_csv(file, sep=' ', header=None)
    answer = train.iloc[:, 0]
    point=train.iloc[:, 1:3].copy()
    D = DistanceMatrix(point)
    D[np.diag_indices_from(D)]=float("inf")
    sort_argD = np.argsort(D, axis=1)  # 排序后的最短邻接点
    # 最短边直接相连
    point["link"] = sort_argD[:,0]
    adj = np.array([point.index, sort_argD[:,0]]).transpose()

    pos = np.array(point.iloc[:, 0:2])
    g.setData(pos=pos, adj=adj,size=0.01, pxMode=False)

    check_mult=np.all(np.tile(adj,(len(adj),1,1))==np.tile(adj[:,::-1],(len(adj),1,1)).swapaxes(0,1),axis=2)
    # row cluster
    tmp = [None] * len(point)
    for i in range(len(point)):
        if tmp[point["link"][i]] == None:
            tmp[i] = tmp[point["link"][i]] = i
        else:
            tmp[i] = tmp[point["link"][i]]
    point["cluster"] = tmp

    adj = np.array(Prim(D))
    g.setData(pos=pos, adj=adj, size=0.01, pxMode=False)

if __name__ == '__main__':
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

