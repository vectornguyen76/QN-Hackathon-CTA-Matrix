import torch.nn.functional as F
import torch
import torch.nn as nn

def loss_classifier(pred_classifier, labels_classifier):
    return nn.BCELoss()(pred_classifier, labels_classifier)

def loss_regressor(pred_regressor, labels_regressor):
    mask = (labels_regressor != 0)
    loss = ((pred_regressor - labels_regressor)**2)[mask].sum() / mask.sum()
    return loss

def loss_softmax(inputs, labels, device):
    mask = (labels != 0)
    # inputs (N, 6, 5)
    n, aspect, rate = inputs.shape
    num = 0
    loss = torch.zeros(labels.shape).to(device)
    for i in range(aspect):
        label_i = labels[:, i].clone()
        label_i[label_i != 0] -= 1
        label_i = label_i.type(torch.LongTensor).to(device)
        loss[:, i] = nn.CrossEntropyLoss(reduction='none')(inputs[:, i, :], label_i)
    loss = loss[mask].sum() / mask.sum()
    return loss

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",):

    # p = torch.sigmoid(inputs)
    p = inputs
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def bce_loss_weights(inputs, targets, weights):
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    weights = targets*(1 / weights.view(1, -1)) + (1 - targets)*(1 / (1 - weights.view(1, -1)))
    loss = ce_loss*weights
    return loss.mean()


def CB_loss(inputs, targets, samples_positive_per_cls, samples_negative_per_cls, no_of_classes=2,loss_type='sigmoid', beta=0.9999, gamma=2):
    samples_per_cls = torch.concat([samples_positive_per_cls.unsqueeze(-1), samples_negative_per_cls.unsqueeze(-1)], dim=-1) # num_cls, 2
    effective_num = 1.0 - torch.pow(beta, samples_per_cls) # num_cls, 2
    weights = (1.0 - beta) / effective_num # num_cls, 2
    weights = weights / weights.sum(dim=-1).reshape(-1, 1) * no_of_classes # num_cls, 2 
    weights = targets*weights[:, 0] + (1 - targets)*weights[:, 1]

    if loss_type == "focal":
        cb_loss = (sigmoid_focal_loss(inputs, targets)*weights).mean()
    elif loss_type == "sigmoid":
        cb_loss = (F.binary_cross_entropy(inputs,targets, reduction="none")*weights).mean()
    return cb_loss