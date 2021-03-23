# Adjusted mutual information

from sklearn import metrics

def AMI(result, label):
    return metrics.adjusted_mutual_info_score(label, result)
