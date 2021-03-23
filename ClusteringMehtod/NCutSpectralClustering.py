import codecs
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time
from Myfunc import Coopcheckdiv
from validation import validation
def load_data(path):
	data_set = list()
	with codecs.open(path) as f:
		for line in f.readlines():
			data = line.strip().split(" ")
			flt_data = list(map(np.double, data))
			data_set.append(flt_data)
	return data_set

def AffMatrix(arrA, arrB,sigma):
    c = sigma
    a = -(np.sum(np.square(np.array(arrA)-np.array(arrB))))
    b = 2 * (np.square(c))
    w = np.exp(a/b)
    return w

def W_Matrix(data):
    dim = data.shape[0]
    createMatrix = np.zeros((dim,dim))
    for i in range(dim):
        x = data[i,:]
        for j in range(dim):
            y = data[j,:]
            createMatrix[i,j] = AffMatrix(x,y,0.5) #计算的高斯核函数放入创建的空矩阵中
    createMatrix = createMatrix - np.identity(dim) # np.identity()创建单位对角矩阵  //  生成对角线为0的矩阵
    return createMatrix

def D_Matrix(W):
    D = np.diag(np.sum(W,axis = 0))
    return D

def NCutSpectralCluster_plot_clusters(data, cluster_assment, method_str):

    plt.figure()
    k = int(max(cluster_assment)) + 1
    colors = [plt.cm.get_cmap("Spectral")(each) for each in np.linspace(0, 1, k)]
    for i, col in zip(range(k), colors):
        per_data_set = data[np.nonzero(cluster_assment == i)]
        plt.plot(per_data_set[:, 0], per_data_set[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=10)
    plt.title(method_str+"Clustering, k = "+str(k), fontsize=15)
    plt.show()

def NCutSpectralClustering(data,k):
    m = data.shape[0]

    W = W_Matrix(data[0:m,:])
    D = np.mat(D_Matrix(W))

    Dn = np.mat(np.diag(np.power(np.sum(W, axis=0), -0.5)))

    W = np.mat(W)
    L = D + W
    
    Ln = Dn * L * Dn
    eigvals, eigvecs = np.linalg.eig(Ln)

    indices = eigvals.argsort()[-k:]
    k_max_eigvecs = eigvecs[:, indices]
    X = np.linalg.norm(k_max_eigvecs, axis=1)

    normalize_eigvecs = np.zeros((m, k))
    
    for i in range(m):
        normalize_eigvecs[i,:] = k_max_eigvecs[i,:] / X[i]
        
    label_predict = KMeans(n_clusters=k).fit_predict(normalize_eigvecs)
    return label_predict
    
if __name__ == '__main__':
    data = np.array(load_data("/Users/kinddle/课程任务/人工智能基础/project_1/five_cluster.txt"))
    m = 2000
    t = time.time_ns()
    label_predict = NCutSpectralClustering(data[0:m,1:],5)

    NCutSpectralCluster_plot_clusters(data[0:m,1:], label_predict, 'spectrum clustering')
    print(f"时间：{(time.time_ns()-t)/1e6}ms")
    check=Coopcheckdiv(label_predict,data[0:m,0])
    print(f"准确率：{check}")
    print("function_name:NCut-SpectralClustering")
    validation(label_predict,data[0:m,0],True)
