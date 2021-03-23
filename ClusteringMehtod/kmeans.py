import codecs
from numpy import *
import matplotlib.pyplot as plt

def load_data(path):
	"""
	@brief      Loads a data
	@param      path  The path
	@return     data set
	"""
	data_set = list()
	with codecs.open(path) as f:
		for line in f.readlines():
			data = line.strip().split(" ")
			flt_data = list(map(double, data))
			data_set.append(flt_data)
	return data_set

def dist_eucl(vecA, vecB):
	"""
	@brief      the similarity function
	@param      vecA  The vector a
	@param      vecB  The vector b
	@return     the euclidean distance
	"""
	return sqrt(sum(power(vecA - vecB, 2)))

def rand_cent(data_mat, k):
	"""
	@brief      select random centroid
	@param      data_mat  The data matrix
	@param      k
	@return     centroids
	"""
	n = shape(data_mat)[1]
	centroids = mat(zeros((k, n)))
	for j in range(n):
		minJ = min(data_mat[:,j]) 
		rangeJ = float(max(data_mat[:,j]) - minJ)
		centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
	return centroids

def kMeans(data_mat, k, dist = "dist_eucl", create_cent = "rand_cent"):
	"""
	@brief      kMeans algorithm
	@param      data_mat     The data matrix
	@param      k            num of cluster
	@param      dist         The distance funtion
	@param      create_cent  The create centroid function
	@return     the cluster
	"""
	m = shape(data_mat)[0]
	# 初始化点的簇
	cluster_assment = mat(zeros((m, 2)))  # 类别，距离
	# 随机初始化聚类初始点
	centroid = eval(create_cent)(data_mat, k)
	cluster_changed = True
	# 遍历每个点
	while cluster_changed:
		cluster_changed = False
		for i in range(m):
			min_index = -1
			min_dist = inf
			for j in range(k):
				distance = eval(dist)(data_mat[i, :], centroid[j, :])
				if distance < min_dist:
					min_dist = distance
					min_index = j
			if cluster_assment[i, 0] != min_index:
				cluster_changed = True
				cluster_assment[i, :] = min_index, min_dist**2
		# 计算簇中所有点的均值并重新将均值作为质心
		for j in range(k):
			per_data_set = data_mat[nonzero(cluster_assment[:,0].A == j)[0]]
			centroid[j, :] = mean(per_data_set, axis = 0)
	return centroid, cluster_assment

def kmeans_plot_cluster(data_mat, cluster_assment, centroid):
	"""
	@brief      plot cluster and centroid
	@param      data_mat        The data matrix
	@param      cluster_assment  The cluste assment
	@param      centroid        The centroid
	@return     
	"""
	plt.figure(figsize=(15, 6), dpi=80)
	plt.subplot(121)
	plt.plot(data_mat[:, 0], data_mat[:, 1], 'o')
	plt.title("source data", fontsize=15)
	plt.subplot(122)
	k = shape(centroid)[0]
	colors = [plt.cm.get_cmap("Spectral")(each) for each in linspace(0, 1, k)]
	for i, col in zip(range(k), colors):
	    per_data_set = data_mat[nonzero(cluster_assment[:,0].A == i)[0]]
	    plt.plot(per_data_set[:, 0], per_data_set[:, 1], 'o', markerfacecolor=tuple(col),
	             markeredgecolor='k', markersize=10)
	for i in range(k):
		plt.plot(centroid[:,0], centroid[:,1], '+', color = 'k', markersize=18)
	plt.title("K-Means Clustering, k = "+str(k), fontsize=15)
	plt.show()

if __name__ == '__main__':
	data_mat = mat(load_data("D:\\大三下\\人工智能基础\\Thecode\\Chapter2\\data\\five_cluster.txt"))
	data_mat = data_mat[:,1:3]
	
	centroid, cluster_assment = kMeans(data_mat, 5)
	sse = sum(cluster_assment[:,1])
	print("SSE is ", sse)
	kmeans_plot_cluster(data_mat, cluster_assment, centroid)


	