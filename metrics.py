from sklearn.metrics import f1_score
import numpy as np

def r2_score(pred, gt):
    result = []
    for i in range(pred.shape[1]):
        a = pred[:,i]
        b = gt[:,i]
        RSS = 0
        K = 0
        for j in range(a.shape[0]):
            RSS += (a[j] - b[j])**2
            K += 16

        result.append(1 - (RSS/K))

    return np.array(result)

def metrics(pred, gt):
    pred = np.array(pred, dtype='float32')
    gt = np.array(gt, dtype='float32')
    
    bool_pred = pred > 0
    bool_gt = gt > 0

    f1s = f1_score(bool_pred, bool_gt, average=None)
    r2s = r2_score(pred, gt)

    result = np.mean(f1s * r2s)
    
    return result

pred = [[1, 0, 3, 4, 2, 0], [0, 2, 2, 5, 1, 0]]
gt = [[1, 1, 3, 5, 3, 1], [0, 0, 3, 4, 0, 0]]

print(metrics(pred, gt))