from sklearn import metrics

def homogeneity(result, label):
    return metrics.homogeneity_score(label, result)

def completeness(result, label):
    return metrics.completeness_score(label, result)

def v_measure(result, label):
    return metrics.v_measure_score(label, result)
