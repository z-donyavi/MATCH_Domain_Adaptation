import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import helpers
from utils import *
import glob
import os

# Results Path
res_path = "results/retraining_exp/"

# Load dataset and algorithm indexes
data_set_index = pd.read_csv("data/data_index.csv", sep=";", index_col="dataset")


# Train/test ratios
train_test_ratios = [np.array([0.5, 0.5])]
train_distributions = {
    2: np.array([[0.5, 0.5]]),
    3: np.array([[0.35, 0.3, 0.35]]),
    4: np.array([[0.25, 0.25, 0.25, 0.25]]),
    5: np.array([[0.2, 0.2, 0.2, 0.2, 0.2]]),
    6: np.array([[0.17, 0.17, 0.17, 0.17, 0.16, 0.16]]),
    7: np.array([[0.15, 0.14, 0.14, 0.14, 0.15, 0.14, 0.14]]),
    10: np.array([[0.1] * 10])
}

# Select datasets with >2 classes
mc_data = data_set_index.loc[data_set_index["classes"] > 2].index
mc_data = mc_data[mc_data != "insects_sound"] 
# mc_data = ['concrete']

results = []
ave = []  # To store averages for each dataset

for dta_name in mc_data:

    n_classes = data_set_index.loc[dta_name, "classes"]
    train_ds = train_distributions[n_classes]
    seed = 4711

    print(f"\nProcessing dataset: {dta_name}")
    X, y, N, Y, n_classes, y_cts, y_idx = helpers.get_xy(dta_name, load_from_disk=False, binned=False)
    test_ds = test_prev(train_ds, n_classes, sample_size=N / 2)

    # Locate the CSV file for this dataset
    base_path = os.getcwd() + '/results/'
    file_pattern = os.path.join(base_path, f"{dta_name}q_details* .csv")
    matching_files = glob.glob(file_pattern)

    if not matching_files:
        print(f"No file found for pattern: {file_pattern}")
        continue

    file_path = matching_files[0]
    print(f"Loading file: {file_path}")
    data = pd.read_csv(file_path)

    for test_idx, test_distr in enumerate(test_ds):
        print(f"\nTest Distribution {test_idx + 1}_{dta_name}: {test_distr}")
        
        # Generate train and test indices
        train_index, test_index, stats_vec = helpers.synthetic_draw(
            N, n_classes, y_cts, y_idx, train_test_ratios[0], train_ds[0], test_distr, seed
        )
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        # Extract row once for this test distribution
        row = data.iloc[test_idx]

        # Initialize result dictionary for this test distribution
        result_row = {"dataset": dta_name, "test_distribution_index": test_idx + 1}

        # Precompute sample weights for all quantifiers
        test_distributions = {
            qun: [row[f"{qun}_Prediction_Class_{i}"] for i in range(n_classes)] for qun in ['EMQ', 'GAC', 'GPAC']
        }


        for qun, test_distribution in test_distributions.items():
            # Calculate sample weights once
            class_weight_ratios = [test_distribution[i] / train_ds[0][i] for i in range(n_classes)]
            sample_weights = np.array([class_weight_ratios[label] for label in y_train])

            # Train Logistic Regression
            # clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000)
            clf = ensemble.RandomForestClassifier()
            clf.fit(X_train, y_train, sample_weight=sample_weights)

            # Predict and evaluate
            predictions = clf.predict(X_test)
            acc = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, average='macro', zero_division=0)
            recall = recall_score(y_test, predictions, average='macro', zero_division=0)
            f1 = f1_score(y_test, predictions, average='macro', zero_division=0)

            # Save metrics
            result_row[f"{qun}_acc"] = acc
            result_row[f"{qun}_precision"] = precision
            result_row[f"{qun}_recall"] = recall
            result_row[f"{qun}_f1_score"] = f1

        # Append the row to results
        results.append(result_row)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate mean values for this dataset
    mean_metrics = results_df.loc[results_df['dataset'] == dta_name].mean(numeric_only=True)
    mean_metrics["dataset"] = dta_name
    ave.append(mean_metrics)

    # Save per-dataset results to a CSV file
    results_df.to_csv(res_path + f"{dta_name}.csv", index=False)
    print(f"Results for {dta_name} saved.")

# Create a summary DataFrame with averages for all datasets
summary_df = pd.DataFrame(ave)
summary_df = summary_df[['dataset'] + [col for col in summary_df.columns if col != 'dataset']]  # Reorder columns

# Save the summary to a final CSV file
summary_df.to_csv(res_path + "All_datasets.csv", index=False)
print("Summary results saved to 'final_summary.csv'.")
