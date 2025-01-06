# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 15:49:22 2022

@author: Zahra
"""

# Model Building functions

import numpy as np
from sklearn import svm, linear_model, ensemble, naive_bayes, discriminant_analysis, model_selection
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import lightgbm
from scipy.stats import mode
import cvxpy as cp
import time


def getScores(X_train, X_test, Y_train, Y_test, nclasses, algs, seed):

    models = [linear_model.LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000),
                  discriminant_analysis.LinearDiscriminantAnalysis(),
                  ensemble.RandomForestClassifier(),
                  svm.SVC(probability=True),
                  lightgbm.LGBMClassifier(),
                  naive_bayes.GaussianNB(),
                  ensemble.GradientBoostingClassifier()]
          
   
    train_scores = np.zeros((len(models), len(X_train), nclasses))
    test_scores = np.zeros((len(models), len(X_test), nclasses))
    Y_cts = np.unique(Y_train, return_counts=True)
    nfolds = min(10, min(Y_cts[1]))
    for i, model in enumerate(models):
        j = 0
        ts_score_fold =  np.zeros((10, len(X_test), nclasses))
        nfolds = min(10, min(Y_cts[1]))
        if nfolds > 1:
            kfold = model_selection.KFold(n_splits=nfolds, random_state=seed, shuffle=True)
            for train, test in kfold.split(X_train, Y_train):
                model.fit(X_train[train], Y_train[train])
                train_scores[i][test] = model.predict_proba(X_train)[test]
                ts_score_fold[j] = model.predict_proba(X_test)
                j +=1
                
            test_scores[i] = np.mean(ts_score_fold, axis=0)
       
        model.fit(X_train, Y_train)

        
        if nfolds < 2:
            train_scores[i] = model.predict_proba(X_train)
            
    return train_scores, test_scores, len(models)


def BUR_MAP(X_test, Y_test, test_scores, p_test, p_train, nmodels, nclasses, classifiers):
    class_labels = list(range(nclasses))    
    adjusted_test_scores = np.zeros((nmodels, len(X_test), nclasses))
    start_time = time.time()
    ratio = p_test/p_train
    for mod in range(nmodels):
        for i in range(len(X_test)):
            for c in range(nclasses):
                adjusted_test_scores[mod, i, c] = (ratio[c] * test_scores[mod,i,c])/(sum([(ratio[j]*test_scores[mod,i,j]) for j in range(nclasses)]))
    
    #classification acc with adjusting Esemble
    predictions_en = np.zeros((len(X_test),nmodels))
    for mod in range(nmodels):
        predictions_en[:,mod] = np.argmax(adjusted_test_scores[mod], axis=1)
    end_time = time.time()    
    adjustment_time = end_time - start_time
    
    final_predictions_en, _ = mode(predictions_en, axis=1, keepdims=True)
    acc_re_en = accuracy_score(Y_test, final_predictions_en)
    prec_re_en = precision_score(Y_test, final_predictions_en, average='macro')
    rec_re_en = recall_score(Y_test, final_predictions_en, average='macro')
    F1_re_en = f1_score(Y_test, final_predictions_en, average='macro')
    
    classifier_names = {
    "LR": predictions_en[:,0],
    "LDA": predictions_en[:,1],
    "RF": predictions_en[:,2],
    "SVM": predictions_en[:,3],
    "LGBM": predictions_en[:,4],
    "NB": predictions_en[:,5],
    "GB": predictions_en[:,6],
    "EN": final_predictions_en
}

    accuracies = []
    precisions = []
    recalls = []
    F1_scores = []
    

    #classification acc with adjusting
    for cl in classifiers:
        # Calculate accuracy for the current classifier
        accuracies.append(accuracy_score(Y_test, classifier_names[cl]))
        precisions.append(precision_score(Y_test, classifier_names[cl], average='macro'))
        recalls.append(recall_score(Y_test, classifier_names[cl], average='macro'))
        F1_scores.append(f1_score(Y_test, classifier_names[cl], average='macro'))
        counts = np.bincount(classifier_names[cl].astype(int), minlength=nclasses)
        test_ratios = counts / np.sum(counts)

    return acc_re_en, prec_re_en, rec_re_en, F1_re_en, accuracies, precisions, recalls, F1_scores, test_ratios




def MATCH(X_test, Y_test, test_scores, p_test, p_train, nmodels, nclasses, classifiers):
    class_labels = list(range(nclasses))
    eps = 1e-10
    num_instances = len(test_scores[0])
    num_classes = nclasses
    num_instances_per_class = np.round(num_instances * np.array(p_test)).astype(int)
    adjusted_priors = np.zeros((nmodels, len(X_test), nclasses))
    count_converge = 0
    count_not_converge = 0
    count_notfeasible = 0
    count_unbounded = 0
    for mod in range(nmodels):
        start_time = time.time()
        log_p_y__x = np.log(test_scores[mod] + eps)  # Using log-probabilities to avoid underflow

        if sum(num_instances_per_class) != num_instances:  # Adjusting rounding differences
            largest_class = np.argmax(num_instances_per_class)
            num_instances_per_class[largest_class] += num_instances - sum(num_instances_per_class)
    
        # Defining binary variables in CVXPY
        a = cp.Variable((num_instances, num_classes), boolean=True)
    
        # Objective: maximize the sum of log-likelihoods
        objective = cp.Maximize(cp.sum(cp.multiply(a, log_p_y__x)))
    
        # Constraints: each instance must be assigned to one class
        constraints = [cp.sum(a, axis=1) == 1]
    
        # Constraints: the number of instances assigned to each class must match the given prevalences
        for j in range(num_classes):
            constraints.append(cp.sum(a[:, j]) == num_instances_per_class[j])
    
        # Define the problem
        problem = cp.Problem(objective, constraints)
        problem.solve()
        end_time = time.time()
        opt_time = end_time-start_time
        
        
        # Check the status of the solver
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print("The solver has converged to an optimal solution.")
            count_converge += 1
        elif problem.status == cp.INFEASIBLE:
            print("The problem is infeasible.")
            count_notfeasible +=1
        elif problem.status == cp.UNBOUNDED:
            print("The problem is unbounded.")
            count_unbounded +=1
        else:
            print("The solver did not converge to an optimal solution.")
            print(f"Solver status: {problem.status}")
            count_not_converge +=1
        # Extracting the solution
        adjusted_priors[mod] = np.round(a.value).astype(int)
    
    print('Converge:', count_converge)
    print('Not_converge:', count_not_converge)
    print('Not_feasible:', count_notfeasible)
    print('Unbounded:', count_unbounded)    
    
    #classification acc with adjusting Esemble
    predictions_en = np.zeros((len(X_test),nmodels))
    for mod in range(nmodels):
        predictions_en[:,mod] = np.argmax(adjusted_priors[mod], axis=1)
    
    final_predictions_en, _ = mode(predictions_en, axis=1, keepdims=True)
    acc_re_en = accuracy_score(Y_test, final_predictions_en)
    prec_re_en = precision_score(Y_test, final_predictions_en, average='macro')
    rec_re_en = recall_score(Y_test, final_predictions_en, average='macro')
    F1_re_en = f1_score(Y_test, final_predictions_en, average='macro')

    classifier_names = {
    "LR": predictions_en[:,0],
    "LDA": predictions_en[:,1],
    "RF": predictions_en[:,2],
    "SVM": predictions_en[:,3],
    "LGBM": predictions_en[:,4],
    "NB": predictions_en[:,5],
    "GB": predictions_en[:,6],
    "EN": final_predictions_en
    }

    accuracies = []
    precisions = []
    recalls = []
    F1_scores = []
    
    for cl in classifiers:
        # Calculate accuracy for the current classifier
        accuracies.append(accuracy_score(Y_test, classifier_names[cl]))
        precisions.append(precision_score(Y_test, classifier_names[cl], average='macro'))
        recalls.append(recall_score(Y_test, classifier_names[cl], average='macro'))
        F1_scores.append(f1_score(Y_test, classifier_names[cl], average='macro'))
        
    # class_index = np.argmax(adjusted_test_scores, axis=1)
    # acc_re = accuracy_score(Y_test, class_index)
    return acc_re_en, prec_re_en, rec_re_en, F1_re_en, accuracies, precisions, recalls, F1_scores, count_converge


def to_str(var):
    if type(var) is list:
        return str(var)[1:-1] # list
    if type(var) is np.ndarray:
        try:
            return str(list(var[0]))[1:-1] # numpy 1D array
        except TypeError:
            return str(list(var))[1:-1] # numpy sequence
    return str(var) # everything else

def _draw_simplex(rn, ndim, min_val, max_trials=100):
     """
     returns a uniform sampling from the ndim-dimensional simplex but guarantees that all dimensions
     are >= min_class_prev (for min_val>0, this makes the sampling not truly uniform)
     :param ndim: number of dimensions of the simplex
     :param min_val: minimum class prevalence allowed. If less than 1/ndim a ValueError will be throw since
     there is no possible solution.
     :return: a sample from the ndim-dimensional simplex that is uniform in S(ndim)-R where S(ndim) is the simplex
     and R is the simplex subset containing dimensions lower than min_val
     """
     if min_val >= 1 / ndim:
         raise ValueError(f'no sample can be draw from the {ndim}-dimensional simplex so that '
                          f'all its values are >={min_val} (try with a smaller value for min_pos)')
     all_u = []
     u_criteria = np.zeros((max_trials, 2))
     trials = 0
     while True:
         u = uniform_prevalence_sampling(rn, ndim)
         if all(u >= min_val):
             return u
         trials += 1
         all_u.append(u)
         if trials >= max_trials:
             for i in range (max_trials):
                 u_criteria[i,0] = sum(all_u[i] < min_val)
                 u_criteria[i,1] = sum(abs(e-min_val) for e in all_u[i] if e < min_val)
                 # u_criteria[i,2] = u_criteria[i,0] * u_criteria[i,1]
             min_ind = np.where(u_criteria[:,1] == u_criteria[:,1].min())
             u = all_u[min_ind[0][0]]
             return u
             # raise ValueError(f'it looks like finding a random simplex with all its dimensions being'
                              # f'>= {min_val} is unlikely (it failed after {max_trials} trials)')
                              
# Supporting Functions (extracted from QuaPy)
def uniform_prevalence_sampling(rn, n_classes, size=1):
    # np.random.seed(4711)
    if n_classes == 2:
        u = np.random.rand(size)
        u = np.vstack([1 - u, u]).T
    else:
        # from https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex
        # u = np.random.rand(size, n_classes - 1)
        u = rn.reshape(1, n_classes - 1)
        u.sort(axis=-1)
        _0s = np.zeros(shape=(size, 1))
        _1s = np.ones(shape=(size, 1))
        a = np.hstack([_0s, u])
        b = np.hstack([u, _1s])
        u = b - a
    if size == 1:
        u = u.flatten()
    return u                              

def test_prev(train_dist, nclasses, sample_size):
 # tr_set = qp.data.base.LabelledCollection(X_train, Y_train)
 np.random.seed(4711)
 # The sizes of samples are the same as the size of training set, proposed by the authors.
 # sample_size = len(tr_set)
 # The number of samples proposed by the authors.
 ensemble_size = 1000
 rn = np.random.rand(ensemble_size, nclasses - 1)
 # Prevalence selection interval [0.05 0.95] of each class, proposed by the authors.
 # min_instances = int(len(tr_set) * 0.05)

 # Generate 30 different distribution vectors.
 prevs = [_draw_simplex(rn[_], ndim=nclasses, min_val=0 ) for _ in range(ensemble_size)]
 
 test_dists = np.array(prevs)

# Calculate the L1 distance between each test distribution and the training distribution
 distances = np.sum(np.abs(test_dists - train_dist), axis=1)
 distances = distances.round(1)
 unique_distances, counts = np.unique(distances, return_counts=True)
 
 return prevs

def compute_rae(p_true, prediction, nclasses, eps):
    p_s = (p_true + eps) / (eps * nclasses + 1)
    phat_s = (prediction + eps) / (eps * nclasses + 1)
    rae = np.abs(phat_s - p_s) / p_s
    return rae