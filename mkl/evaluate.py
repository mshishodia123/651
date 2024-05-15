import math
# from mkl import validation
from sklearn.svm import SVC
import torch
import numpy as np


def margin(K,Y, return_coefs=False, init_vals=None, solver='cvxopt', max_iter=-1, tol=1e-6):
    '''margin optimization with libsvm'''

    Y = torch.tensor([1 if y == Y[0] else -1 for y in Y])
    params = {'K': K, 'Y': Y, 'init_vals': init_vals, 'max_iter': max_iter, 'tol': tol}

    svm = SVC(C=1e7, kernel='precomputed', tol=tol, max_iter=max_iter).fit(K,Y)
    n = len(Y)
    gamma = torch.zeros(n).double()
    gamma[svm.support_] = torch.tensor(svm.dual_coef_)
    idx_pos = gamma > 0
    idx_neg = gamma < 0
    sum_pos, sum_neg = gamma[idx_pos].sum(), gamma[idx_neg].sum()
    gamma[idx_pos] /= sum_pos
    gamma[idx_neg] /= sum_neg
    gammay = gamma * Y
    obj = (gammay.view(n,1).T @ K @ gammay).item() **.5
    return obj, gamma


def trace(K, Y=None):
    # K = validation.check_K(K)
    return K.diag().sum().item()

def frobenius(K):
    # K = validation.check_K(K)
    return ( (K**2).sum()**.5 ).item()

def kernel_similarity(K1, K2):
    # K1 = validation.check_K(K1)
    # K2 = validation.check_K(K2)
    # Q = K1.flatten() @ K2.flatten()
    Q = math.sqrt(trace(K1@(K2.T)))

    Q = Q/math.sqrt(frobenius(K1) * frobenius(K2))

    return Q

def spectral_ratio(K, Y=None, norm=True):
    # K = validation.check_K(K)
    n = K.size()[0]
    c = trace(K)/frobenius(K)
    return (c-1)/(n**.5 - 1) if norm else c




