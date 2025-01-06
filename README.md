# MATCH: A Maximum-Likelihood Approach for Classification under Label Shift

MATCH is a domain adaptation framework designed for robust evaluation under varying conditions. This repository provides an implementation that allows users to experiment with different algorithms, datasets, quantifiers, and classifiers.


In the following, we describe how one can reproduce our results.


## Experiment

This experiment has been run with Python 3.9.11. We used some packages like ```numpy```, ```pandas```, ```scikit-learn``` and ```cvxpy``` are used in most algorithms.


### Main Experiments

#### MATCH
To reproduce all our experiments with all datasets, all of quantifiers (EMQ, GAC, GPACC), and classifiers (LR, LDA, RF, SVM, LGBM, NB, GB), one can simply run our main script via 

```bash
    python3 -m main.py -a {algorithms} -d {datasets} --cl{classifier}
```

where quantifiers and datasets to run on can be specified by their respective names as listed in ```alg_index.csv``` and ```data/data_index.csv```. When none of the arguments are specified, all experiments will be executed. 

#### BUR-MAP
To run **BUR-MAP**, follow these steps:

1. Open the `main.py` script.
2. Replace the name `MATCH` with `BUR_MAP` at the following lines:
   - **Line 316**
   - **Line 363**

After making these changes, you can run the code in the same way as for **MATCH** using the following command:

```bash
python3 -m main.py -a {algorithm} -d {dataset} --cl {classifier}

#### Retraining Experiments

After completing the main experiments, you can run the retraining_exp.py script. This script processes the raw results to calculate the training and test ratios, which are then used for retraining purposes.





