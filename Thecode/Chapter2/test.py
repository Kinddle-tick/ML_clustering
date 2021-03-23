from ClusteringEvaluation.purity import *
from ClusteringEvaluation.NMI import *
from ClusteringEvaluation.AMI import *
from ClusteringEvaluation.RI import *
from ClusteringEvaluation.ARI import *
from ClusteringEvaluation.Fmeasure import *
from ClusteringEvaluation.Vmeasure import *
from ClusteringEvaluation.FowlkesMallows import *
from ClusteringMehtod.kmeans import *
from ClusteringMehtod.dbscan import *
from ClusteringMehtod.kmeanspp import *
from ClusteringMehtod.plot_clusters import *
from numpy import *
import matplotlib.pyplot as plt
import time

def validation(result,label,prints=False):
    rtndic={}
    purity_ = purity(result, label)
    rtndic["purity"]=purity_

    NMI_ = NMI(result, label)
    rtndic["NMI_"] = NMI_

    AMI_ = AMI(result, label)
    rtndic["AMI"] = AMI_

    RI_ = RI(result, label)
    rtndic["RI"] = RI_

    ARI_ = ARI(result, label)
    rtndic["ARI"] = ARI_

    F_measure_ = F_measure(result, label)
    rtndic["F_measure"] = F_measure_

    fowlkes_mallows_ = fowlkes_mallows(result, label)
    rtndic["fowlkes_mallows"] = fowlkes_mallows_

    homogeneity_=homogeneity(result, label)
    rtndic["homogeneity_"] = homogeneity

    completeness_ = completeness(result, label)
    rtndic["completeness"] = completeness_

    v_measure_ = v_measure(result, label)
    rtndic["v_measure"] = v_measure_

    if prints:
        print("DBSCAN evaluation:")
        print("Purity:" + str(purity_))
        print("NMI:" + str(NMI_))
        print("AMI:" + str(AMI_))
        print("RI:" + str(RI_))
        print("ARI:" + str(ARI_))
        print("F-measure:" + str(F_measure_))
        print("Fowlkes Mallows:" + str(fowlkes_mallows_))
        print("homogeneity:" + str(homogeneity_))
        print("completeness:" + str(completeness_))
        print("v_measure:" + str(v_measure_))

    return rtndic

data_mat = mat(load_data("data/five_cluster.txt"))

# dbscan
t= time.time_ns()
IDX = DBSCAN(data_mat[:, 1:3], 0.4, 9)
print((time.time_ns()-t)*1e-6)
result= IDX
label = data_mat[:, 0].T.tolist()[0]
print("DBSCAN evaluation:")
print("Purity:" + str(purity(result, label)))
print("NMI:" + str(NMI(result, label)))
print("AMI:" + str(AMI(result, label)))
print("RI:" + str(RI(result, label)))
print("ARI:" + str(ARI(result, label)))
print("F-measure:" + str(F_measure(result, label)))
print("Fowlkes Mallows:" + str(fowlkes_mallows(result, label)))
print("homogeneity:" + str(homogeneity(result, label)))
print("completeness:" + str(completeness(result, label)))
print("v_measure:" + str(v_measure(result, label)))

plot_clusters(data_mat[:, 1:3],IDX,"DBSCAN ")

# k-means
centroid, cluster_assment = kMeans(data_mat[:, 1:3], 5)
label = data_mat[:, 0].T.tolist()[0]
result = cluster_assment[:, 0].T.tolist()[0]
print("k-means evaluation:")
print("Purity:" + str(purity(result, label)))
print("NMI:" + str(NMI(result, label)))
print("AMI:" + str(AMI(result, label)))
print("RI:" + str(RI(result, label)))
print("ARI:" + str(ARI(result, label)))
print("F-measure:" + str(F_measure(result, label)))
print("Fowlkes Mallows:" + str(fowlkes_mallows(result, label)))
print("homogeneity:" + str(homogeneity(result, label)))
print("completeness:" + str(completeness(result, label)))
print("v_measure:" + str(v_measure(result, label)))

kmeans_plot_cluster(data_mat[:, 1:3],cluster_assment,centroid)

# k-means++
centroid, cluster_assment = kpp_Means(data_mat[:, 1:3], 5)
label = data_mat[:, 0].T.tolist()[0]
result = cluster_assment[:, 0].T.tolist()[0]
print("k-means++ evaluation:\n")
print("Purity:" + str(purity(result, label)))
print("NMI:" + str(NMI(result, label)))
print("AMI:" + str(AMI(result, label)))
print("RI:" + str(RI(result, label)))
print("ARI:" + str(ARI(result, label)))
print("F-measure:" + str(F_measure(result, label)))
print("Fowlkes Mallows:" + str(fowlkes_mallows(result, label)))
print("homogeneity:" + str(homogeneity(result, label)))
print("completeness:" + str(completeness(result, label)))
print("v_measure:" + str(v_measure(result, label)))

kpp_Means_plot_cluster(data_mat[:, 1:3], cluster_assment, centroid)
