import numpy as np

class ScalarMetric():
    def __init__(self):
        self.scalar = 0
        self.num = 0
    def update(self, scalar):
        self.scalar += scalar
        self.num += 1
        return self
    def compute(self):
        return self.scalar / self.num
    def reset(self):
        self.scalar = 0
        self.num = 0

class AccuracyMetric():
    def __init__(self):
        self.correct = 0
        self.num = 0
    def update(self, y_pred, y_true):
        self.correct += (y_pred == y_true).sum()
        self.num += len(y_pred)*y_pred.shape[1]
    def compute(self):
        return self.correct / self.num
    def reset(self):
        self.correct = 0
        self.num = 0

def precision(y_pred, y_true):
    true_positive = np.logical_and(y_pred, y_true).sum(axis=0)
    false_positive = np.logical_and(y_pred, np.logical_not(y_true)).sum(axis=0)
    return true_positive / (true_positive + false_positive)

def recall(y_pred, y_true):
    true_positive = np.logical_and(y_pred, y_true).sum(axis=0)
    false_negative = np.logical_and(np.logical_not(y_pred), y_true).sum(axis=0)
    return true_positive / (true_positive + false_negative)

class F1_score():
    def __init__(self):
        self.y_pred = None
        self.y_true = None
    def update(self, y_pred, y_true):
        self.y_pred = np.concatenate([self.y_pred, y_pred], axis=0) if self.y_pred is not None else y_pred
        self.y_true = np.concatenate([self.y_true, y_true], axis=0) if self.y_true is not None else y_true
    def compute(self):
        f1_score = np.zeros(self.y_pred.shape[1])
        precision_score = precision(self.y_pred != 0, self.y_true != 0)
        recall_score = recall(self.y_pred != 0, self.y_true != 0)
        mask_precision_score = np.logical_and(precision_score != 0, np.logical_not(np.isnan(precision_score)))
        mask_recall_score = np.logical_and(recall_score != 0, np.logical_not(np.isnan(recall_score)))
        mask = np.logical_and(mask_precision_score, mask_recall_score)
        print("Precision:",precision_score)
        print("Recall", recall_score)
        f1_score[mask] = 2* (precision_score[mask] * recall_score[mask]) / (precision_score[mask] + recall_score[mask])
        return f1_score

class R2_score():
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def update(self, y_pred, y_true):
        self.y_pred = np.concatenate([self.y_pred, y_pred], axis=0) if self.y_pred is not None else y_pred
        self.y_true = np.concatenate([self.y_true, y_true], axis=0) if self.y_true is not None else y_true
    
    def compute(self):
        mask = np.logical_and(self.y_pred !=0, self.y_true != 0)
        rss = (((self.y_pred - self.y_true)**2)*mask).sum(axis=0) 
        k = (mask*16).sum(axis=0)
        r2_score = np.ones(rss.shape[0])
        mask2 = (k != 0)
        r2_score[mask2] = 1 - rss[mask2]/k[mask2]
        return r2_score