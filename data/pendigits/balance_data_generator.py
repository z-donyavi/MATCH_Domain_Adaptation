import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import cvxpy as cp
import os
import time
from sklearn.utils import resample

# Load the dataset
file_path = 'pendigits.csv'
data = pd.read_csv(file_path)

# Ensure the last column is the class label (modify if needed)
class_column = data.columns[-1]
classes = data[class_column].unique()

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
 ensemble_size = 50
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

def balance_classes(data, total_instances, class_distribution, seed=42, enforce_classes=None):
    """
    Balance the dataset for a given number of total instances and classes.

    Parameters:
    - data: DataFrame containing the dataset.
    - total_instances: Total number of instances to sample.
    - class_distribution: List specifying the proportion of instances for each class.
    - seed: Random seed for reproducibility.
    - enforce_classes: List of classes to enforce in the output. If None, uses classes from `class_distribution`.

    Returns:
    - DataFrame with balanced classes.
    """
    np.random.seed(seed)
    balanced_data = []
    num_classes = len(class_distribution)
    
    # Use enforce_classes if specified, otherwise use classes from data
    target_classes = enforce_classes if enforce_classes else classes[:num_classes]
    
    for i, cls in enumerate(target_classes):
        class_data = data[data[class_column] == cls]
        n_samples = int(total_instances * class_distribution[i])
        resampled_class_data = resample(class_data, n_samples=n_samples, replace=True, random_state=seed)
        balanced_data.append(resampled_class_data)

    return pd.concat(balanced_data)


# Global parameters
global_seeds = [4711]
classifiers = ['LR']
algs = ['GPAC']

res_path = "results/"
os.makedirs(res_path, exist_ok=True)  # Ensure the results path exists

# Experiment configurations
instance_range = range(1000, 10001, 1000)
num_classes_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]  # Different configurations for number of classes
# num_classes_list = [2]  # Different configurations for number of classes

# test_distributions = {
#     2: [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5]],
#     3: [[0.1, 0.1, 0.8], [0.2, 0.2, 0.6], [0.3, 0.3, 0.4], [0.4, 0.4, 0.2], [0.33, 0.33, 0.34]],
#     4: [[0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.25, 0.25], [0.4, 0.3, 0.2, 0.1], [0.1, 0.3, 0.3, 0.3], [0.2, 0.2, 0.3, 0.3]],
#     5: [[0.15, 0.1, 0.65, 0.1, 0], [0.45, 0.1, 0.3, 0.05, 0.1], [0.2, 0.25, 0.25, 0.1, 0.2],
#         [0.35, 0.05, 0.05, 0.05, 0.5], [0.05, 0.25, 0.15, 0.15, 0.4]],
#     6: [[0.15, 0.1, 0.55, 0.1, 0, 0.1], [0.4, 0.1, 0.25, 0.05, 0.1, 0.1], [0.2, 0.2, 0.2, 0.1, 0.2, 0.1],
#         [0.35, 0.05, 0.05, 0.05, 0.05, 0.45], [0.05, 0.25, 0.15, 0.15, 0.1, 0.3]],
#     7: [[0.1, 0.1, 0.1, 0.5, 0.1, 0, 0.1], [0.4, 0.1, 0.2, 0.05, 0.1, 0.1, 0.05], [0.15, 0.2, 0.15, 0.1, 0.2, 0.1, 0.1],
#         [0.3, 0.05, 0.05, 0.05, 0.05, 0.05, 0.45], [0.05, 0.25, 0.1, 0.15, 0.1, 0.3, 0.05]],
#     8: [[0.1, 0.1, 0.1, 0.3, 0.1, 0.1, 0.1, 0.1], [0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1],
#         [0.15, 0.1, 0.1, 0.15, 0.15, 0.15, 0.1, 0.1], [0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1],
#         [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]],
#     9: [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
#         [0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15], [0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.1],
#         [0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111]],
#     10: [[0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0, 0.1, 0.05, 0.05], [0.2, 0.05, 0.15, 0.05, 0.1, 0.15, 0.05, 0.05, 0.1, 0.1],
#         [0, 0.1, 0.05, 0.1, 0.05, 0.1, 0.1, 0.15, 0.15, 0.2], [0.05, 0.05, 0.05, 0.35, 0.15, 0.05, 0, 0.1, 0.1, 0.1],
#         [0.05, 0.1, 0.1, 0.15, 0.1, 0.15, 0.05, 0.1, 0.1, 0.1]]
# }

results = []

for num_instances in instance_range:
    for num_classes in num_classes_list:
        train_distribution = [1 / num_classes] * num_classes  # Uniform distribution for training
        
        # Determine target classes for this experiment
        target_classes = sorted(classes[:num_classes])  # Ensure consistent order
        
        # Generate training data
        train_data = balance_classes(data, num_instances, train_distribution, seed=42, enforce_classes=target_classes)
        label_to_index = {label: idx for idx, label in enumerate(target_classes)}
        train_data[class_column] = train_data[class_column].map(label_to_index)
        print("Training Data Distribution:")
        print(train_data[class_column].value_counts(normalize=True))
        
        test_dists = test_prev(train_distribution, num_classes, sample_size=num_instances)

        
        for test_distribution in test_dists:
            print(f"\nProcessing: {num_instances} instances with {num_classes} classes and test distribution {test_distribution}")
            
            # Generate test data
            test_data = balance_classes(data, num_instances, test_distribution, seed=42, enforce_classes=target_classes)
            test_data[class_column] = test_data[class_column].map(label_to_index)
            print("Test Data Distribution:")
            print(test_data[class_column].value_counts(normalize=True))

            X_train = train_data.drop(columns=[class_column]).values
            y_train = train_data[class_column].values
            X_test = test_data.drop(columns=[class_column]).values
            y_test = test_data[class_column].values


            # Get test scores (example with Logistic Regression)
            clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000)
            clf.fit(X_train, y_train)
            train_scores = clf.predict_proba(X_train)
            test_scores = clf.predict_proba(X_test)

            # Measure retraining time
            # label_to_index = {label: idx for idx, label in enumerate(sorted(np.unique(y_train)))}
            start_ret = time.time()
            class_weight_ratios = [test_distribution[i] / train_distribution[i] for i in range(num_classes)]
            sample_weights = np.array([class_weight_ratios[label] for label in y_train])
            
            clf.fit(X_train, y_train, sample_weight=sample_weights)
            clf.predict(X_test)
            retraining_time = time.time() - start_ret

            # Measure adjustment time
            start_bur = time.time()
            ratio = np.array(test_distribution) / np.array(train_distribution)
            adjusted_test_scores = np.zeros_like(test_scores)

            for i in range(len(X_test)):
                for c in range(num_classes):
                    adjusted_test_scores[i, c] = (ratio[c] * test_scores[i, c]) / (
                        sum([(ratio[j] * test_scores[i, j]) for j in range(num_classes)])
                    )
            prediction = np.argmax(adjusted_test_scores, axis=1)        
            adjustment_time = time.time() - start_bur

            # Optimization (Matching)
            start_time = time.time()
            eps = 1e-10
            num_instances_per_class = np.round(len(X_test) * np.array(test_distribution)).astype(int)
            log_p_y__x = np.log(test_scores + eps)

            if sum(num_instances_per_class) != len(X_test):
                largest_class = np.argmax(num_instances_per_class)
                num_instances_per_class[largest_class] += len(X_test) - sum(num_instances_per_class)

            a = cp.Variable((len(X_test), num_classes), boolean=True)
            objective = cp.Maximize(cp.sum(cp.multiply(a, log_p_y__x)))
            constraints = [cp.sum(a, axis=1) == 1]
            for j in range(num_classes):
                constraints.append(cp.sum(a[:, j]) == num_instances_per_class[j])

            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            # Check optimization success
            if problem.status == cp.OPTIMAL:
                print("Optimization was successful.")
            elif problem.status == cp.INFEASIBLE:
                print("Optimization failed: problem is infeasible.")
            elif problem.status == cp.UNBOUNDED:
                print("Optimization failed: problem is unbounded.")
            else:
                print(f"Optimization ended with status: {problem.status}")
            
            adjusted_priors = np.round(a.value).astype(int)
            prediction = np.argmax(adjusted_priors, axis=1)  

            opt_time = time.time() - start_time

            # Save results
            result_row = {
                "instances": num_instances,
                "classes": num_classes,
                "test_distribution": test_distribution,
                "retraining_time": retraining_time,
                "adjustment_time": adjustment_time,
                "opt_time": opt_time
            }
            results.append(result_row)

# Save overall results
results_df = pd.DataFrame(results)
results_path = os.path.join(res_path, "Overall_timing_50.csv")
results_df.to_csv(results_path, index=False)
print(f"Overall timing summary saved to '{results_path}'.")
