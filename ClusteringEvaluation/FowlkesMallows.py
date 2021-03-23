from sklearn import metrics
def fowlkes_mallows(result, label):
    return metrics.fowlkes_mallows_score(label, result)
