# MATCH: A Maximum-Likelihood Approach for Classification under Label Shift

MATCH is a domain adaptation framework designed for robust evaluation under varying conditions. This repository provides an implementation that allows users to experiment with different algorithms, datasets, quantifiers, and classifiers.


In the following, we describe how one can reproduce our results.


## Experiment

This experiment has been run with Python 3.9.11. We used some packages like ```numpy```, ```pandas```, ```scikit-learn``` and ```cvxpy``` are used in most algorithms.


##### Loading Datasets

Each dataset has a ```prep.py``` file to prepare. After preparing all datasets, the parameter ```load_from_disk=False``` in line 167 in ```run.py``` can be set to the ```True``` value.


#### Main Experiments

### MATCH
To reproduce all our experiments with all datasets, all of quantifiers (EMQ, GAC, GPACC), and classifiers (LR, LDA, RF, SVM, LGBM, NB, GB), one can simply run our main script via 

```bash
    python3 -m main.py -a {algorithms} -d {datasets} --cl{classifier}
```

where algorithms and datasets to run on can be specified by their respective names as listed in ```alg_index.csv``` and ```data/data_index.csv```. When none of the arguments are specified, all experiments will be executed. 








