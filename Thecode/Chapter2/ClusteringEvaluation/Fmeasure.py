
def contingency_table(result, label):
    
    total_num = len(label)
    
    TP = TN = FP = FN = 0
    for i in range(total_num):
        for j in range(i + 1, total_num):
            if label[i] == label[j] and result[i] == result[j]:
                TP += 1
            elif label[i] != label[j] and result[i] != result[j]:
                TN += 1
            elif label[i] != label[j] and result[i] == result[j]:
                FP += 1
            elif label[i] == label[j] and result[i] != result[j]:
                FN += 1
    return (TP, TN, FP, FN)

def precision(result, label):
    TP, TN, FP, FN = contingency_table(result, label)
    return 1.0*TP/(TP + FP)
def recall(result, label):
    TP, TN, FP, FN = contingency_table(result, label)
    return 1.0*TP/(TP + FN)
def F_measure(result, label, beta=1):
    prec = precision(result, label)
    r = recall(result, label)
    return (beta * beta + 1) * prec * r / (beta * beta * prec + r)
    