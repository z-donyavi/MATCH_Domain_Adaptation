# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 16:59:03 2022

@author: Zahra
"""

import numpy as np
import cvxpy as cvx
from sklearn import metrics
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances, cosine_distances, haversine_distances, pairwise_distances
import quadprog
import quapy as qp
from scipy.stats import wasserstein_distance
from metric_learn import LMNN, NCA, LFDA, MLKR




def apply_fusion(fusionMethod, predictions, Axis = 0):
    if fusionMethod == "amean":
        return np.mean(predictions, Axis)
    if fusionMethod == "median":
        return np.median(predictions, Axis)
    if fusionMethod == "min":
        return np.min(predictions, Axis)
    if fusionMethod == "max":
        return np.max(predictions, Axis)
    if fusionMethod == "prod":
        return np.prod(predictions, Axis)
    
    return np.median(predictions, Axis)

def apply_fusion(fusionMethod, predictions, Axis = 0):
    if fusionMethod == "amean":
        return np.mean(predictions, Axis)
    if fusionMethod == "median":
        return np.median(predictions, Axis)
    if fusionMethod == "min":
        return np.min(predictions, Axis)
    if fusionMethod == "max":
        return np.max(predictions, Axis)
    if fusionMethod == "prod":
        return np.prod(predictions, Axis)
    
    return np.median(predictions, Axis)

def apply_fusion_one_step(predictions, fusionMethod, Axis = 0):
     if fusionMethod == "amean":
         return np.mean(predictions, axis=Axis)
     if fusionMethod == "median":
         return np.median(predictions, Axis)
     if fusionMethod == "min":
         return np.min(predictions, Axis)
     if fusionMethod == "max":
         return np.max(predictions, Axis)
     if fusionMethod == "prod":
         return np.prod(predictions, Axis)

def One_step_fusion(quantifiers_list, tr_scores, te_scores, train_labels, nmodels, nclasses, step):
    
    nquant = len(quantifiers_list)
   
    pred_prop = []
    pred_prop_array = np.zeros((nmodels*nquant,nclasses))
    for quantifier in quantifiers_list:
        pred = Apply_One_step_fusion(quantifier, tr_scores, te_scores, train_labels, nmodels, nclasses, step)
        pred_prop.append(pred)
    
    for i in range(nquant):
        pred_prop_array[i*nmodels:(i+1)*nmodels,:] = pred_prop[i]

    return pred_prop_array

def apply_Fusion_quantifierEnsemble(fusionQuant,qntMethod, tr_scores, te_scores, tr_labels, nmodels, nclasses, step):

    if qntMethod == "EM":
        return EnsembleEM(te_scores, tr_labels, nmodels, nclasses, fusionQuant, step)
    if qntMethod == "GACC":
        return EnsembleGAC(tr_scores, te_scores, tr_labels, nmodels, nclasses, fusionQuant, step)
    if qntMethod == "GPACC":
        return EnsembleGPAC(tr_scores, te_scores, tr_labels, nmodels, nclasses, fusionQuant, step)
    if qntMethod == "FM":
        return EnsembleFM(tr_scores, te_scores, tr_labels, nmodels, nclasses, fusionQuant, step)
    
def Apply_One_step_fusion(qntMethod, tr_scores, te_scores, tr_labels, nmodels, nclasses, step):
    if qntMethod == "EM":
        return EnsembleEM(te_scores, tr_labels, nmodels, nclasses, 'none', step)
    if qntMethod == "GACC":
        return EnsembleGAC(tr_scores, te_scores, tr_labels, nmodels, nclasses, 'none', step)
    if qntMethod == "GPACC":
        return EnsembleGPAC(tr_scores, te_scores, tr_labels, nmodels, nclasses, 'none', step)
    if qntMethod == "FM":
        return EnsembleFM(tr_scores, te_scores, tr_labels, nmodels, nclasses, 'none', step)
    if qntMethod == "EDy":
        return EnsembleEDy(tr_scores, te_scores, tr_labels, nmodels, nclasses, 'none', step)
    
# Supporting functions

class Distances(object):
    
    def __init__(self,P,Q):
        if sum(P)<1e-20 or sum(Q)<1e-20:
            raise "One or both vector are zero (empty)..."
        if len(P)!=len(Q):
            raise "Arrays need to be of equal sizes..."
        #use numpy arrays for efficient coding
        P=np.array(P,dtype=float);Q=np.array(Q,dtype=float)
        #Correct for zero values
        P[np.where(P<1e-20)]=1e-20
        Q[np.where(Q<1e-20)]=1e-20
        self.P=P
        self.Q=Q
        
    def sqEuclidean(self):
        P=self.P; Q=self.Q; 
        return np.sum((P-Q)**2)
    def probsymm(self):
        P=self.P; Q=self.Q; 
        return 2*np.sum((P-Q)**2/(P+Q))
    def topsoe(self):
        P=self.P; Q=self.Q
        return np.sum(P*np.log(2*P/(P+Q))+Q*np.log(2*Q/(P+Q)))
    def hellinger(self):
        P=self.P; Q=self.Q
        return np.sqrt(np.sum((np.sqrt(P) - np.sqrt(Q))**2))
    def kl_divergence(self):
        P = self.P
        Q = self.Q
        return np.sum(P * np.log(P / Q))
    def js_divergence(self):
        P = self.P
        Q = self.Q
        M = 0.5 * (P + Q)
        return 0.5 * (self.kl_divergence() + Distances(Q, M).kl_divergence())
    def bhattacharyya_distance(self):
        P = self.P
        Q = self.Q
        return -np.log(np.sum(np.sqrt(P * Q)))
    def total_variation_distance(self):
        P = self.P
        Q = self.Q
        return 0.5 * np.sum(np.abs(np.array(P) - np.array(Q)))
    def earth_movers_distance(self):
        P = self.P
        Q = self.Q
        emd = wasserstein_distance(P, Q)
        return emd


def distance(sc_1, sc_2, measure):
    dist = Distances(sc_1, sc_2)
    if measure == 'sqEuclidean':
        return dist.sqEuclidean()
    if measure == 'topsoe':
        return dist.topsoe()
    if measure == 'probsymm':
        return dist.probsymm()
    if measure == 'hellinger':
        return dist.hellinger()
    if measure == 'kl_divergence':
        return dist.kl_divergence()
    if measure == 'js_divergence':
        return dist.js_divergence()
    if measure == 'bhattacharyya_distance':
        return dist.bhattacharyya_distance()
    if measure == 'total_variation_distance':
        return dist.total_variation_distance()
    if measure == 'earth_movers_distance':
        return dist.earth_movers_distance()
    
    print("Error, unknown distance specified, returning topsoe")
    return dist.topsoe()

def distance_matrix(mat_1, mat_2, measure):
    dist_matrix = np.zeros((mat_1.shape[0], mat_2.shape[0]))
    for i, row_1 in enumerate(mat_1):
        for j, row_2 in enumerate(mat_2):
            # dist_matrix[i, j] = distance(row_1, row_2, measure)
            dist_matrix[i, j] = measure(row_1, row_2)
    return dist_matrix


def TernarySearch(left, right, f, eps=1e-4):

    while True:
        if abs(left - right) < eps:
            return (left + right) / 2, f((left + right) / 2)
    
        leftThird  = left + (right - left) / 3
        rightThird = right - (right - left) / 3
    
        if f(leftThird) > f(rightThird):
            left = leftThird
        else:
            right = rightThird 
            
def LinearSearch(left, right, f, eps=1e-4):
    
    p = np.linspace(left, right, num=21, endpoint=True)
    p[0] += 0.01
    p[-1] -= 0.01
    selected_prev = p[0]
    for prev in p:
        if f(prev)<f(selected_prev):
            selected_prev = prev
    return selected_prev, f(selected_prev)

def getHist(scores, nbins):
    breaks = np.linspace(0, 1, int(nbins)+1)
    breaks = np.delete(breaks, -1)
    breaks = np.append(breaks,1.1)
    
    re = np.repeat(1/(len(breaks)-1), (len(breaks)-1))  
    for i in range(1,len(breaks)):
        re[i-1] = (re[i-1] + len(np.where((scores >= breaks[i-1]) & (scores < breaks[i]))[0]) ) / (len(scores)+1)
    return re

def class_dist(Y, nclasses):
    return np.array([np.count_nonzero(Y == i) for i in range(nclasses)]) / Y.shape[0]


# Base quantifiers

#EMQ quapy

def EM_quapy(test_scores, train_labels, nclasses):
    epsilon=1e-4
    posterior_probabilities = test_scores 
    tr_prev = class_dist(train_labels, nclasses)
    MAX_ITER = 1000
    """
    Computes the `Expectation Maximization` routine.

    :param tr_prev: array-like, the training prevalence
    :param posterior_probabilities: `np.ndarray` of shape `(n_instances, n_classes,)` with the
        posterior probabilities
    :param epsilon: float, the threshold different between two consecutive iterations
        to reach before stopping the loop
    :return: a tuple with the estimated prevalence values (shape `(n_classes,)`) and
        the corrected posterior probabilities (shape `(n_instances, n_classes,)`)
    """
    Px = posterior_probabilities
    Ptr = np.copy(tr_prev)
    qs = np.copy(Ptr)  # qs (the running estimate) is initialized as the training prevalence

    s, converged = 0, False
    qs_prev_ = None
    while not converged and s < MAX_ITER:
        # E-step: ps is Ps(y|xi)
        ps_unnormalized = (qs / Ptr) * Px
        ps = ps_unnormalized / ps_unnormalized.sum(axis=1, keepdims=True)

        # M-step:
        qs = ps.mean(axis=0)

        if qs_prev_ is not None and qp.error.mae(qs, qs_prev_) < epsilon and s > 10:
            converged = True

        qs_prev_ = qs
        s += 1

    if not converged:
        print('[warning] the method has reached the maximum number of iterations; it might have not converged')

    return qs

import numpy as np

import numpy as np

def CC(test_scores, n_classes):
    # Get the predictions by taking the argmax of the test scores
    predictions = np.argmax(test_scores, axis=1)
    
    # Initialize an array of zeros for all classes
    test_prev = np.zeros(n_classes)
    
    # Get the counts of the predicted classes
    unique, counts = np.unique(predictions, return_counts=True)
    
    # Ensure counts is a flat array
    counts = counts.flatten()
    
    # Update the test_prev array with the counts
    for cls, count in zip(unique, counts):
        test_prev[cls] = count
    
    # Normalize to get the proportions
    test_prev = test_prev / test_prev.sum()
    
    return test_prev


    
def EMQ(test_scores, train_labels, nclasses):
    max_it = 1000        # Max num of iterations
    eps = 1e-6           # Small constant for stopping criterium

    p_tr = class_dist(train_labels, nclasses)
    p_s = np.copy(p_tr)
    p_cond_tr = np.array(test_scores)
    p_cond_s = np.zeros(p_cond_tr.shape)

    for it in range(max_it):
        r = p_s / p_tr
        p_cond_s = p_cond_tr * r
        s = np.sum(p_cond_s, axis = 1)
        for c in range(nclasses):
            p_cond_s[:,c] = p_cond_s[:,c] / s
        p_s_old = np.copy(p_s)
        p_s = np.sum(p_cond_s, axis = 0) / p_cond_s.shape[0]
        if (np.sum(np.abs(p_s - p_s_old)) < eps):
            break

    return(p_s/np.sum(p_s))

def EMQ_ini(test_scores, train_labels, nclasses, ts_prev):
    ts_prev = ts_prev/sum(ts_prev)
    epsilon=1e-4
    tr_prev = class_dist(train_labels, nclasses)
    MAX_ITER = 1000
    
    
    adjusted_test_scores = np.zeros((len(test_scores), nclasses))
    ratio = ts_prev/tr_prev
    for i in range(len(test_scores)):
        for c in range(nclasses):
            adjusted_test_scores[i, c] = (ratio[c] * test_scores[i,c])/(sum([(ratio[j]*test_scores[i,j]) for j in range(nclasses)]))
    """
    Computes the `Expectation Maximization` routine.

    :param tr_prev: array-like, the training prevalence
    :param posterior_probabilities: `np.ndarray` of shape `(n_instances, n_classes,)` with the
        posterior probabilities
    :param epsilon: float, the threshold different between two consecutive iterations
        to reach before stopping the loop
    :return: a tuple with the estimated prevalence values (shape `(n_classes,)`) and
        the corrected posterior probabilities (shape `(n_instances, n_classes,)`)
    """
    Px = adjusted_test_scores
    Ptr = np.copy(ts_prev)
    qs = np.copy(Ptr)  # qs (the running estimate) is initialized as the training prevalence

    s, converged = 0, False
    qs_prev_ = None
    while not converged and s < MAX_ITER:
        # E-step: ps is Ps(y|xi)
        ps_unnormalized = (qs / Ptr) * Px
        ps = ps_unnormalized / ps_unnormalized.sum(axis=1, keepdims=True)

        # M-step:
        qs = ps.mean(axis=0)

        if qs_prev_ is not None and qp.error.mae(qs, qs_prev_) < epsilon and s > 10:
            converged = True

        qs_prev_ = qs
        s += 1

    if not converged:
        print('[warning] the method has reached the maximum number of iterations; it might have not converged')

    return qs


def GAC(train_scores, test_scores, train_labels, nclasses):
   
    yt_hat = np.argmax(train_scores, axis = 1)
    y_hat = np.argmax(test_scores, axis = 1)
    CM = metrics.confusion_matrix(train_labels, yt_hat, normalize="true").T
    p_y_hat = np.zeros(nclasses)
    values, counts = np.unique(y_hat, return_counts=True)
    p_y_hat[values] = counts 
    p_y_hat = p_y_hat/p_y_hat.sum()
    
    p_hat = cvx.Variable(CM.shape[1])
    constraints = [p_hat >= 0, cvx.sum(p_hat) == 1.0]
    problem = cvx.Problem(cvx.Minimize(cvx.norm(CM @ p_hat - p_y_hat)), constraints)
    problem.solve()
    return p_hat.value

def GPAC(train_scores, test_scores, train_labels, nclasses):

    CM = np.zeros((nclasses, nclasses))
    for i in range(nclasses):
        idx = np.where(train_labels == i)[0]
        CM[i] = np.sum(train_scores[idx], axis=0)
        CM[i] /= np.sum(CM[i])
    CM = CM.T
    p_y_hat = np.sum(test_scores, axis = 0)
    p_y_hat = p_y_hat / np.sum(p_y_hat)
    
    p_hat = cvx.Variable(CM.shape[1])
    constraints = [p_hat >= 0, cvx.sum(p_hat) == 1.0]
    problem = cvx.Problem(cvx.Minimize(cvx.norm(CM @ p_hat - p_y_hat)), constraints)
    problem.solve()
    return p_hat.value

def FM(train_scores, test_scores, train_labels, nclasses):

    CM = np.zeros((nclasses, nclasses))
    y_cts = np.array([np.count_nonzero(train_labels == i) for i in range(nclasses)])
    p_yt = y_cts / train_labels.shape[0]
    for i in range(nclasses):
        idx = np.where(train_labels == i)[0]
        CM[:, i] += np.sum(train_scores[idx] > p_yt, axis=0) 
    CM = CM / y_cts
    p_y_hat = np.sum(test_scores > p_yt, axis = 0) / test_scores.shape[0]
    
    p_hat = cvx.Variable(CM.shape[1])
    constraints = [p_hat >= 0, cvx.sum(p_hat) == 1.0]
    problem = cvx.Problem(cvx.Minimize(cvx.norm(CM @ p_hat - p_y_hat)), constraints)
    problem.solve()
    return p_hat.value

def dpofa(m):
    r = np.array(m, copy=True)
    n = len(r)
    for k in range(n):
        s = 0.0
        if k >= 1:
            for i in range(k):
                t = r[i, k]
                if i > 0:
                    t = t - np.sum(r[0:i, i] * r[0:i, k])
                t = t / r[i, i]
                r[i, k] = t
                s = s + t * t
        s = r[k, k] - s
        if s <= 0.0:
            return k+1, r
        r[k, k] = np.sqrt(s)
    return 0, r


def nearest_pd(A):
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if is_pd(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    indendity_matrix = np.eye(A.shape[0])
    k = 1
    while not is_pd(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += indendity_matrix * (-mineig * k ** 2 + spacing)
        k += 1

    return A3


def is_pd(m):
    return dpofa(m)[0] == 0


def solve_ed(G, a, C, b):
    sol = quadprog.solve_qp(G=G, a=a, C=C, b=b)
    prevalences = sol[0]
    # the last class was removed from the problem, its prevalence is 1 - the sum of prevalences for the other classes
    return np.append(prevalences, 1 - prevalences.sum())


def compute_ed_param_train(distance_func, train_distrib, classes, n_cls_i):
    n_classes = len(classes)
    #  computing sum de distances for each pair of classes
    K = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        K[i, i] = distance_func(train_distrib[classes[i]], train_distrib[classes[i]]).sum()
        for j in range(i + 1, n_classes):
            K[i, j] = distance_func(train_distrib[classes[i]], train_distrib[classes[j]]).sum()
            K[j, i] = K[i, j]

    #  average distance
    K = K / np.dot(n_cls_i, n_cls_i.T)

    B = np.zeros((n_classes - 1, n_classes - 1))
    for i in range(n_classes - 1):
        B[i, i] = - K[i, i] - K[-1, -1] + 2 * K[i, -1]
        for j in range(n_classes - 1):
            if j == i:
                continue
            B[i, j] = - K[i, j] - K[-1, -1] + K[i, -1] + K[j, -1]

    #  computing the terms for the optimization problem
    G = 2 * B
    if not is_pd(G):
        G = nearest_pd(G)

    C = -np.vstack([np.ones((1, n_classes - 1)), -np.eye(n_classes - 1)]).T
    b = -np.array([1] + [0] * (n_classes - 1), dtype=float)

    return K, G, C, b


def compute_ed_param_test(distance_func, train_distrib, test_distrib, K, classes, n_cls_i):

    n_classes = len(classes)
    Kt = np.zeros(n_classes)
    for i in range(n_classes):
        Kt[i] = distance_func(train_distrib[classes[i]], test_distrib).sum()

    Kt = Kt / (n_cls_i.squeeze() * float(len(test_distrib)))

    a = 2 * (- Kt[:-1] + K[:-1, -1] + Kt[-1] - K[-1, -1])
    return a

def EDy_opt(tr_scores, labels, te_scores, nclasses):
    lmnn = LMNN(k=5, learn_rate=1e-6)
    lmnn.fit(tr_scores, labels)
    
    # nca = NCA(max_iter=1000)
    # nca.fit(tr_scores, labels)
    
    # lfda = MLKR()
    # lfda.fit(tr_scores, labels)
    
    # Transform the scores using the learned metric
    tr_scores_transformed = lmnn.transform(tr_scores)
    te_scores_transformed = lmnn.transform(te_scores)
    
    # Use the transformed scores for distance calculations
    distance = pairwise_distances
    # distance = lmnn.get_metric()
    
    # distance = 'hellinger'
    # distance = manhattan_distances
    # distance = euclidean_distances
    # distance = cosine_distances
    classes_ = np.unique(labels)
    train_distrib_ = dict.fromkeys(classes_)
    train_n_cls_i_ = np.zeros((nclasses, 1))
        
    if len(labels) == len(tr_scores_transformed):
            y_ext_ = labels
    else:
            y_ext_ = np.tile(labels, len(tr_scores_transformed) // len(labels))
        
    for n_cls, cls in enumerate(classes_):
        train_distrib_[cls] = tr_scores_transformed[y_ext_ == cls,:]
        train_n_cls_i_[n_cls, 0] = len(train_distrib_[cls])
        
    K_, G_, C_, b_ = compute_ed_param_train(distance, train_distrib_, classes_, train_n_cls_i_)
      
    a_ = compute_ed_param_test(distance, train_distrib_, te_scores_transformed, K_, classes_, train_n_cls_i_)

    prevalences = solve_ed(G=G_, a=a_, C=C_, b=b_)

    return prevalences/np.sum(prevalences)


def EDy(tr_scores, labels, te_scores, nclasses):
    distance = manhattan_distances
    classes_ = np.unique(labels)
    train_distrib_ = dict.fromkeys(classes_)
    train_n_cls_i_ = np.zeros((nclasses, 1))
        
    if len(labels) == len(tr_scores):
            y_ext_ = labels
    else:
            y_ext_ = np.tile(labels, len(tr_scores) // len(labels))
        
    for n_cls, cls in enumerate(classes_):
        train_distrib_[cls] = tr_scores[y_ext_ == cls,:]
        train_n_cls_i_[n_cls, 0] = len(train_distrib_[cls])
        
    K_, G_, C_, b_ = compute_ed_param_train(distance, train_distrib_, classes_, train_n_cls_i_)
      
    a_ = compute_ed_param_test(distance, train_distrib_, te_scores, K_, classes_, train_n_cls_i_)

    prevalences = solve_ed(G=G_, a=a_, C=C_, b=b_)

    return prevalences/np.sum(prevalences)

# Ensemble quantifiers


def EnsembleEM(test_scores, train_labels, nmodels, nclasses, fusionMethod, step):
    p_hat = np.zeros((nmodels, nclasses))
    for m in range(nmodels):
        p_hat[m] = EMQ(test_scores[m], train_labels, nclasses)
    if step == 'Two':
        p = apply_fusion(fusionMethod, p_hat, 0)
        p = p/np.sum(p)
    else:
        p = p_hat
    return(p)

def EnsembleGAC(train_scores, test_scores, train_labels, nmodels, nclasses, fusionMethod, step):
    p_hat = np.zeros((nmodels, nclasses))
    for m in range(nmodels):
        p_hat[m] = GAC(train_scores[m], test_scores[m], train_labels, nclasses)
    
    if step == 'Two':
        p = apply_fusion(fusionMethod, p_hat, 0)
        p = p/np.sum(p)
    else:
        p = p_hat
    return(p)

def EnsembleGPAC(train_scores, test_scores, train_labels, nmodels, nclasses, fusionMethod, step):
    p_hat = np.zeros((nmodels, nclasses))
    for m in range(nmodels):
        p_hat[m] = GPAC(train_scores[m], test_scores[m], train_labels, nclasses)
    
    if step == 'Two':
        p = apply_fusion(fusionMethod, p_hat, 0)
        p = p/np.sum(p)
    else:
        p = p_hat
    return(p)

def EnsembleFM(train_scores, test_scores, train_labels, nmodels, nclasses, fusionMethod, step):
    p_hat = np.zeros((nmodels, nclasses))
    for m in range(nmodels):
        p_hat[m] = FM(train_scores[m], test_scores[m], train_labels, nclasses)
    
    if step == 'Two':
        p = apply_fusion(fusionMethod, p_hat, 0)
        p = p/np.sum(p)
    else:
        p = p_hat
    return(p)

def EnsembleEDy(train_scores, test_scores, train_labels, nmodels, nclasses, fusionMethod, step):
    p_hat = np.zeros((nmodels, nclasses))
    for m in range(nmodels):
        p_hat[m] = EDy(train_scores[m], train_labels, test_scores[m], nclasses)
    if step == 'Two':
        p = apply_fusion(fusionMethod, p_hat, 0)
        p = p/np.sum(p)
    else:
        p = p_hat
    return(p)

## KDEy

from sklearn.base import BaseEstimator
from sklearn.neighbors import KernelDensity
from quapy.data import *
from quapy.method.aggregative import AggregativeSoftQuantifier
import quapy.functional as F


class KDEBase:
    """
    Common ancestor for KDE-based methods. Implements some common routines.
    """

    BANDWIDTH_METHOD = ['scott', 'silverman']

    @classmethod
    def _check_bandwidth(cls, bandwidth):
        """
        Checks that the bandwidth parameter is correct

        :param bandwidth: either a string (see BANDWIDTH_METHOD) or a float
        :return: nothing, but raises an exception for invalid values
        """
        assert bandwidth in KDEBase.BANDWIDTH_METHOD or isinstance(bandwidth, float), \
            f'invalid bandwidth, valid ones are {KDEBase.BANDWIDTH_METHOD} or float values'
        if isinstance(bandwidth, float):
            assert 0 < bandwidth < 1,  "the bandwith for KDEy should be in (0,1), since this method models the unit simplex"

    def get_kde_function(self, X, bandwidth):
        """
        Wraps the KDE function from scikit-learn.

        :param X: data for which the density function is to be estimated
        :param bandwidth: the bandwidth of the kernel
        :return: a scikit-learn's KernelDensity object
        """
        return KernelDensity(bandwidth=bandwidth).fit(X)

    def pdf(self, kde, X):
        """
        Wraps the density evalution of scikit-learn's KDE. Scikit-learn returns log-scores (s), so this
        function returns :math:`e^{s}`

        :param kde: a previously fit KDE function
        :param X: the data for which the density is to be estimated
        :return: np.ndarray with the densities
        """
        return np.exp(kde.score_samples(X))

    def get_mixture_components(self, X, y, n_classes, bandwidth):
        """
        Returns an array containing the mixture components, i.e., the KDE functions for each class.

        :param X: the data containing the covariates
        :param y: the class labels
        :param n_classes: integer, the number of classes
        :param bandwidth: float, the bandwidth of the kernel
        :return: a list of KernelDensity objects, each fitted with the corresponding class-specific covariates
        """
        return [self.get_kde_function(X[y == cat], bandwidth) for cat in range(n_classes)]



class KDEyML(AggregativeSoftQuantifier, KDEBase):
    """
    Kernel Density Estimation model for quantification (KDEy) relying on the Kullback-Leibler divergence (KLD) as
    the divergence measure to be minimized. This method was first proposed in the paper
    `Kernel Density Estimation for Multiclass Quantification <https://arxiv.org/abs/2401.00490>`_, in which
    the authors show that minimizing the distribution mathing criterion for KLD is akin to performing
    maximum likelihood (ML).

    The distribution matching optimization problem comes down to solving:

    :math:`\\hat{\\alpha} = \\arg\\min_{\\alpha\\in\\Delta^{n-1}} \\mathcal{D}(\\boldsymbol{p}_{\\alpha}||q_{\\widetilde{U}})`

    where :math:`p_{\\alpha}` is the mixture of class-specific KDEs with mixture parameter (hence class prevalence)
    :math:`\\alpha` defined by

    :math:`\\boldsymbol{p}_{\\alpha}(\\widetilde{x}) = \\sum_{i=1}^n \\alpha_i p_{\\widetilde{L}_i}(\\widetilde{x})`

    where :math:`p_X(\\boldsymbol{x}) = \\frac{1}{|X|} \\sum_{x_i\\in X} K\\left(\\frac{x-x_i}{h}\\right)` is the
    KDE function that uses the datapoints in X as the kernel centers.

    In KDEy-ML, the divergence is taken to be the Kullback-Leibler Divergence. This is equivalent to solving:
    :math:`\\hat{\\alpha} = \\arg\\min_{\\alpha\\in\\Delta^{n-1}} -
    \\mathbb{E}_{q_{\\widetilde{U}}} \\left[ \\log \\boldsymbol{p}_{\\alpha}(\\widetilde{x}) \\right]`

    which corresponds to the maximum likelihood estimate.

    :param classifier: a sklearn's Estimator that generates a binary classifier.
    :param val_split: specifies the data used for generating classifier predictions. This specification
        can be made as float in (0, 1) indicating the proportion of stratified held-out validation set to
        be extracted from the training set; or as an integer (default 5), indicating that the predictions
        are to be generated in a `k`-fold cross-validation manner (with this integer indicating the value
        for `k`); or as a collection defining the specific set of data to use for validation.
        Alternatively, this set can be specified at fit time by indicating the exact set of data
        on which the predictions are to be generated.
    :param bandwidth: float, the bandwidth of the Kernel
    :param n_jobs: number of parallel workers
    :param random_state: a seed to be set before fitting any base quantifier (default None)
    """

    def __init__(self, classifier: BaseEstimator, val_split=10, bandwidth=0.1, n_jobs=None, random_state=None):
        self._check_bandwidth(bandwidth)
        self.classifier = classifier
        self.val_split = val_split
        self.bandwidth = bandwidth
        self.n_jobs = n_jobs
        self.random_state=random_state

    def aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):
        self.mix_densities = self.get_mixture_components(*classif_predictions.Xy, data.n_classes, self.bandwidth)
        return self

    def aggregate(self, posteriors: np.ndarray):
        """
        Searches for the mixture model parameter (the sought prevalence values) that maximizes the likelihood
        of the data (i.e., that minimizes the negative log-likelihood)

        :param posteriors: instances in the sample converted into posterior probabilities
        :return: a vector of class prevalence estimates
        """
        np.random.RandomState(self.random_state)
        epsilon = 1e-10
        n_classes = len(self.mix_densities)
        test_densities = [self.pdf(kde_i, posteriors) for kde_i in self.mix_densities]

        def neg_loglikelihood(prev):
            test_mixture_likelihood = sum(prev_i * dens_i for prev_i, dens_i in zip (prev, test_densities))
            test_loglikelihood = np.log(test_mixture_likelihood + epsilon)
            return  -np.sum(test_loglikelihood)

        return F.optim_minimize(neg_loglikelihood, n_classes)
    
# import pymc3 as pm
# import theano.tensor as tt
# import numpy as np

# def EDy_bayesian(tr_scores, labels, te_scores, nclasses, priors):
#     with pm.Model() as model:
#         # Define priors for the class prevalences in the test set
#         prevalences = pm.Dirichlet('prevalences', a=priors['alpha'], shape=nclasses)
        
#         # Define the likelihood of the observed data
#         # Assuming tr_scores are the conditional probabilities from the training set
#         # and labels are the observed class counts in the training set
#         likelihood = pm.Multinomial('likelihood', n=labels.sum(), p=tr_scores, observed=labels)
        
#         # The objective is to find the test prevalences that minimize the distance
#         # between the test predictions (te_scores) and the training conditional probabilities (tr_scores)
#         # scaled by the unknown test prevalences
#         test_likelihood = pm.Multinomial('test_likelihood', n=te_scores.shape[0], p=tt.dot(prevalences, tr_scores), observed=te_scores)
        
#         # Perform the Bayesian update to get the posterior
#         trace = pm.sample(1000, tune=500)
        
#         # Use the posterior to estimate the prevalences in the test set
#         ppc = pm.sample_posterior_predictive(trace, var_names=['prevalences'])
        
#         # Calculate the prevalence estimates for the test set
#         test_prevalences = ppc['prevalences'].mean(axis=0)
        
#         return test_prevalences

# # Example usage:
# # priors should be a dictionary with key 'alpha', representing the concentration parameters of the Dirichlet distribution
# priors = {
#     'alpha': np.array([your_alpha])  # Replace with your actual concentration parameters
# }

# # Call the function with your data
# test_prevalences = EDy_bayesian(tr_scores, labels, te_scores, nclasses, priors)
    