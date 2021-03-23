import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D






if __name__ == "__main__":
    file_list = ["two_cluster.txt"]

    for file in file_list:
        train = pd.read_csv(file, sep=' ',header=None)
        x = train.iloc[:, 0]
        y = train.iloc[:, 1]
        z = train.iloc[:, 2]
        lens = len(x)
        div = len(set(x))+1
        train.iloc[200:, [1, 2]] -= 1
        train.to_csv("two_c.txt",header=0,index=0,sep=' ')
        print("saved")
        c = np.row_stack((x/div,[0.5]*lens,[0.5]*lens)).transpose()
        # plt.title(fig,fontsize='large',fontweight='bold')
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(x,y,z)
        plt.savefig(f"fig/3d_point_{file[:-4]}.jpeg")
        # plt.cla()
        figs = plt.figure()
        plt.scatter(y,z,c=c)
        plt.savefig(f"fig/2d_point_{file[:-4]}.jpeg")
        # plt.cla()
