# Adjusted rand index

from sklearn import metrics

def ARI(result, label):
    return metrics.adjusted_rand_score(label, result)
