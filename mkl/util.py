
import torch


def process(X):
    if type(X) != torch.Tensor:
        X = torch.tensor(X)
    X = X.type(torch.float64)
    return X


def summation (KL, weights = None):
    l = len(KL)
    weights = torch.ones(l) if weights is None else weights

    K = process(KL[0]) * weights[0]
    for i in range(1, l):
        K = K + weights[i] * KL[i]
    return K


def average (KL, weights = None):
    l = len(KL)
    weights = torch.ones(l) if weights is None else weights
    K = summation(KL, weights) / torch.sum(weights)
    return K

def identity_kernel(n):
    return torch.diag(torch.ones(n,  dtype=torch.double))




