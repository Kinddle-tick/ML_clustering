import collections
import math

def NMI(result, label):
    total_num = len(label)
    cluster_counter = collections.Counter(result)
    original_counter = collections.Counter(label)
    
    MI = 0
    eps = 1.4e-45 # 避免log 0
    
    for k in cluster_counter:
        for j in original_counter:
            count = 0
            for i in range(len(result)):
                if result[i] == k and label[i] == j:
                    count += 1
            p_k = 1.0*cluster_counter[k] / total_num
            p_j = 1.0*original_counter[j] / total_num
            p_kj = 1.0*count / total_num
            MI += p_kj * math.log(p_kj /(p_k * p_j) + eps, 2)
    
    # calculate NMI
    H_k = 0
    for k in cluster_counter:
        H_k -= (1.0*cluster_counter[k] / total_num) * math.log(1.0*cluster_counter[k] / total_num+eps, 2)
    H_j = 0
    for j in original_counter:
        H_j -= (1.0*original_counter[j] / total_num) * math.log(1.0*original_counter[j] / total_num+eps, 2)
        
    return 2.0 * MI / (H_k + H_j)