import numpy as np
import matplotlib.pyplot as plt

def plot_clusters(data, cluster_assment, method_str):

    plt.figure()
    # plt.figure(figsize=(15, 6), dpi=80)
    # plt.subplot(121)
    # plt.plot(data[:, 0], data[:, 1], 'o')
    # plt.title("source data", fontsize=15)

    # plt.subplot(122)
    k = int(max(cluster_assment)) + 1
    colors = [plt.cm.get_cmap("Spectral")(each) for each in np.linspace(0, 1, k)]
    for i, col in zip(range(k), colors):
	    per_data_set = data[np.nonzero(cluster_assment == i)]
	    plt.plot(per_data_set[:, 0], per_data_set[:, 1], 'o', markerfacecolor=tuple(col),
	             markeredgecolor='k', markersize=10)
    plt.title(method_str+"Clustering, k = "+str(k), fontsize=15)
    plt.show()
    