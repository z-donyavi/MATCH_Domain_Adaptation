import argparse
import numpy as np
import pandas as pd
import helpers
import os
from time import localtime, strftime
from utils import *
from helper_algs import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import mode


# ==============================================================================
# Global Variables
# ==============================================================================
res_path = "results/"
# global data set index
data_set_index = pd.read_csv("data/data_index.csv",
                             sep=";",
                             index_col="dataset")

# global algorithm index
algorithm_index = pd.read_csv("alg_index.csv",
                              sep=";",
                              index_col="algorithm")


algorithms = list(algorithm_index.index)


global_seeds = [4711]



# train/test ratios to test against
train_test_ratios = [[0.5, 0.5]]

train_test_ratios = [np.array(d) for d in train_test_ratios]

train_distributions = dict()

train_distributions[2] = np.array([[0.5, 0.5]])
train_distributions[3] = np.array([[0.35, 0.3, 0.35]])
train_distributions[4] = np.array([[0.25, 0.25, 0.25, 0.25]])
train_distributions[5] = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])
train_distributions[6] = np.array([[0.17, 0.17, 0.17, 0.17, 0.16, 0.16]])
train_distributions[7] = np.array([[0.15, 0.14, 0.14, 0.14, 0.15, 0.14, 0.14]])
train_distributions[10] = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])

mc_data = data_set_index.loc[data_set_index.loc[:, "classes"] > 2].index


def parse_args():
    """Parses arguments given to script

    Returns:
        dict-like object -- Dict-like object containing all given arguments
    """
    parser = argparse.ArgumentParser()

    # Test parameters
    parser.add_argument(
        "-a", "--algorithms", nargs="+", type=str,
        choices=algorithms, default=algorithms,
        help="Algorithms used in evaluation."
    )
    parser.add_argument(
        "-d", "--datasets", nargs="*", type=str,
        default=None,
        help="Datasets used in evaluation."
    )
    parser.add_argument(
        "--mc", type=int, default=None,
        help="Whether or not to run multiclass experiments"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=global_seeds,
        help="Seeds to be used in experiments. By default, all seeds will be used."
    )
    parser.add_argument(
        "--dt", type=int, nargs="+", default=None,
        help="Index for train/test-splits to be run."
    )
    parser.add_argument(
        "--cl", "--classifiers", nargs="*", type=str,
        default=None,
        help="Classifiers used in the experiments."
    )
    return parser.parse_args()


def run_synth(data_sets=None,
              algs=None,
              dt_index=None,
              b_cl=None,
              b_mc=None,
              seeds=global_seeds):
    
    if data_sets is None:
        df_ind = data_set_index
    else:
        df_ind = data_set_index.loc[data_sets]

    if dt_index is None:
        dt_ratios = train_test_ratios
    else:
        dt_ratios = [train_test_ratios[i] for i in dt_index]
        
    if b_cl is None:
        classifiers = ['LR','LDA', 'RF', 'SVM', 'LGBM', 'NB', 'GB']
    else:
        classifiers = b_cl    

    if b_mc is None:
        data_sets = list(df_ind.index)
    if b_mc == 0:
        df_ind = df_ind.loc[df_ind["classes"] == 2]
        data_sets = list(df_ind.index)
    if b_mc == 1:
        df_ind = df_ind.loc[df_ind["classes"] > 2]
        df_ind = df_ind.drop(df_ind[df_ind['abbr'] == 'insecs'].index)
        data_sets = list(df_ind.index)

    if algs is None:
        alg_ind = algorithms
    else:
        alg_ind = algs


    mean_classification_matrices = []
    for dta_name in data_sets:

        n_classes = df_ind.loc[dta_name, "classes"]

        # build training and test class distributions
        train_ds = train_distributions[n_classes]
        columns = [al for al in algs]
        columns.insert(0, 'seed')
        All_AEs = pd.DataFrame(columns = columns)
        classification_outputs = []


        for seed in seeds:

            # ----run on unbinned data -----------------------

               Mean_AEs, step, classification_matrix = data_synth_experiments(dta_name, binned=False, algs=alg_ind, dt_ratios=dt_ratios, train_ds=train_ds,
                                       classifiers=classifiers,seed=seed)
               All_AEs = All_AEs.append(pd.Series(Mean_AEs, index=All_AEs.columns[:len(Mean_AEs)]), ignore_index=True)
               classification_outputs.append(classification_matrix)
               
        
        concatenated_array = np.vstack(classification_outputs)
        column = ['L1_dist', 'NO_adj_EN', 'Oracle_EN'] 
        for name in classifiers:
            column.append(f"NO_adj_{name}_acc")
            for cl in range(n_classes):
                column.append(f"NO_adj_{name}_prec_cl_{cl}")
            for cl in range(n_classes):    
                column.append(f"NO_adj_{name}_rec_cl_{cl}")
            for cl in range(n_classes):    
                column.append(f"NO_adj_{name}_F1_cl_{cl}")
                
        for name in classifiers:        
            column.append(f"Oracle_{name}_acc")
            for cl in range(n_classes):
                column.append(f"Oracle_{name}_prec_cl_{cl}")
            for cl in range(n_classes):    
                column.append(f"Oracle_{name}_rec_{cl}")
            for cl in range(n_classes):    
                column.append(f"Oracle_{name}_F1_{cl}")
                
        for name in algs:
            column.append(f"adj_EN_by_{name}")
            for cls_name in classifiers: 
                column.append(f"adj_{cls_name}_by_{name}_acc")
                for cl in range(n_classes):
                    column.append(f"adj_{cls_name}_by_{name}_prec_cl_{cl}")
                for cl in range(n_classes):
                    column.append(f"adj_{cls_name}_by_{name}_rec_cl_{cl}")
                for cl in range(n_classes):
                    column.append(f"adj_{cls_name}_by_{name}_F1_cl_{cl}")
                    
        column += ['Q Error ' + al for al in algs]
        Classification_ACC = pd.DataFrame(concatenated_array, columns=column)
        fnamereS = res_path + dta_name + "_Classification_ACC_" + strftime("%Y-%m-%d_%H-%M-%S", localtime()) + "_dir_MAE_LR " + ".csv"
        Classification_ACC.to_csv(fnamereS, index=False, sep=',')
        
        # Calculate mean of classification matrix for each dataset
        mean_classification_matrix = Classification_ACC.mean(axis=0)
        mean_classification_matrix = mean_classification_matrix.drop('L1_dist')  # Drop the first column mean
        mean_classification_matrices.append(mean_classification_matrix)
        
        fnameres = res_path + dta_name + "_All_AEs_" + strftime("%Y-%m-%d_%H-%M-%S", localtime()) + "_dir_MAE_LR " + ".csv"           
        All_AEs.to_csv(fnameres, index=False, sep=',')

    # Concatenate mean classification matrices for all datasets
    final_mean_classification_matrix = pd.concat(mean_classification_matrices, axis=1)
    
    # Transpose the DataFrame to have datasets as rows and metrics as columns
    final_mean_classification_matrix = final_mean_classification_matrix.T
    final_mean_classification_matrix.insert(0, 'Dataset Name', data_sets)
    # Save mean classification matrix for all datasets to csv file
    file_name = res_path + "All_Datasets_Mean_Classification_Matrix_" + strftime("%Y-%m-%d_%H-%M-%S", localtime()) + "_dir_MAE_LR " + ".csv"
    final_mean_classification_matrix.to_csv(file_name, index=False, sep=',')
        
        

def data_synth_experiments(
        dta_name,
        binned,
        algs,
        dt_ratios,
        train_ds,
        classifiers,
        seed=4711):
    if len(algs) == 0 or len(dt_ratios) == 0 or len(train_ds) == 0:
        return

    print(dta_name)
    X, y, N, Y, n_classes, y_cts, y_idx = helpers.get_xy(dta_name, load_from_disk=False, binned=binned)

    fusion_list = ['median']
    test_ds = test_prev(train_ds, n_classes, sample_size=N/2)
    n_combs = len(dt_ratios) * len(train_ds) * len(test_ds)
    n_cols = 5 + 4 * n_classes + n_classes * len(algs)+ ((n_classes+1) * len(algs))

    stats_matrix = np.zeros((n_combs, n_cols))
    classification_matrix = np.zeros((n_combs, 4+len(classifiers)+len(algs)+((3*n_classes)+len(classifiers))*2 +((3*n_classes+len(classifiers))*len(algs)*len(classifiers))))
    
    i = 0
    m = 0
    n = 0
    convergence_count_oracle = 0 
    convergence_count_predict = 0

    first_loop = True
    test_ratio_cls_adj = []
    for test_distr in test_ds:
        index_offset = 1
      
        print('Training and Test dists:')
        print(dt_ratios)
        print(train_ds)
        print(test_distr)

        train_index, test_index, stats_vec = helpers.synthetic_draw(N, n_classes, y_cts, y_idx, dt_ratios[0],
                                                                    train_ds[0], test_distr, seed)
        
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        
        #get train and test scores from a bag of classifires
        
        score_path = os.getcwd() + '/Classification/' + dta_name + '/' + str(seed) + '/' + str(dt_ratios[0]) + '/' + str(train_ds[0]) + '/' + str(test_distr[0:2]) + '/'                    

        if os.path.exists(score_path):
            train_scores = np.load(score_path + 'train_scores.npy')
            test_scores = np.load(score_path + 'test_scores.npy')
        else:
            train_scores, test_scores, nmodels = getScores(X_train, X_test, y_train, y_test, n_classes, algs, global_seeds[0])
            os.makedirs(score_path)
            np.save(os.path.join(score_path, 'train_scores.npy'), train_scores)
            np.save(os.path.join(score_path, 'test_scores.npy'), test_scores)
            
        
        
        j = len(stats_vec)
        stats_matrix[i, 0:j] = stats_vec
        nmodels = 7
        
        tr_scores = train_scores
        te_scores = test_scores
        
        #classification acc without adjusting 

        predictions_en = np.zeros((len(X_test),nmodels))
        for mod in range(nmodels):
            predictions_en[:,mod] = np.argmax(test_scores[mod], axis=1)
            # acc_cls_before_adj.append(accuracy_score(y_test, predictions_en[:,mod]))
        
        final_predictions_en, _ = mode(predictions_en, axis=1, keepdims=True)
        acc_en_before_adj = accuracy_score(y_test, final_predictions_en)
        
        classifier_names = {
        "LR": predictions_en[:,0],
        "LDA": predictions_en[:,1],
        "RF": predictions_en[:,2],
        "SVM": predictions_en[:,3],
        "LGBM": predictions_en[:,4],
        "NB": predictions_en[:,5],
        "GB": predictions_en[:,6]
    }
        classifier_numbers = {
        "LR": 0,
        "LDA": 1,
        "RF": 2,
        "SVM": 3,
        "LGBM": 4,
        "NB": 5,
        "GB": 6
        }
        
        acc_cls_before_adj = []
        precisions_cls_before_adj = []
        racalls_cls_before_adj = []
        F1s_cls_before_adj = []
        #classifiers acc without adjusting
        for cl_s in classifiers:
            acc_cls_before_adj.append(accuracy_score(y_test, classifier_names[cl_s]))
            precisions_cls_before_adj.append(precision_score(y_test, classifier_names[cl_s], average=None))
            racalls_cls_before_adj.append(recall_score(y_test, classifier_names[cl_s], average=None))
            F1s_cls_before_adj.append(f1_score(y_test, classifier_names[cl_s], average=None))
        
        #Oracle
        acc_oracle_en, prec_oracle_en, rec_oracle_en, F1_oracle_en, acc_cls_oracle,  cls_precisions_oracle, cls_recalls_oracle, cls_F1s_oracle, conv_oracle\
            = MATCH(X_test, y_test, te_scores, test_distr, train_ds[0], 7, n_classes, classifiers)
        convergence_count_oracle = convergence_count_oracle + conv_oracle    
        l1_distance = np.sum(np.abs(train_ds[0] - test_distr))
        l1_distance = l1_distance.round(1)
        classification_matrix[i,0]=l1_distance
        classification_matrix[i,1]=acc_en_before_adj
        classification_matrix[i,2]=acc_oracle_en
        index_offset = 3
        for num in range(len(classifiers)):
            classification_matrix[i, index_offset] = acc_cls_before_adj[num]
            index_offset += 1
            classification_matrix[i, index_offset:index_offset+n_classes] = precisions_cls_before_adj[num]
            index_offset += n_classes
            classification_matrix[i, index_offset:index_offset+n_classes] = racalls_cls_before_adj[num]
            index_offset += n_classes
            classification_matrix[i, index_offset:index_offset+n_classes] = F1s_cls_before_adj[num]
            index_offset += n_classes
            classification_matrix[i, index_offset] = acc_cls_oracle[num]
            index_offset += 1
            classification_matrix[i, index_offset:index_offset+n_classes] = cls_precisions_oracle[num]
            index_offset += n_classes
            classification_matrix[i, index_offset:index_offset+n_classes] = cls_recalls_oracle[num]
            index_offset += n_classes
            classification_matrix[i, index_offset:index_offset+n_classes] = cls_F1s_oracle[num]
            index_offset += n_classes

        for cl, str_alg in enumerate(algs):
            with open(dta_name+'.txt', 'a') as f:
                print(str_alg, file=f)
                print(str_alg)
            if str_alg == 'MCMQ':
                quantifiers_list = ['EM', 'GACC', 'GPACC', 'FM']
                for fusion in fusion_list:
                     outputs = One_step_fusion(quantifiers_list, tr_scores, te_scores, y_train, nmodels, n_classes, 'One')
                     p = apply_fusion_one_step(outputs, fusion)
                     p = p/np.sum(p)
                     stats_matrix[i, j:(j + n_classes)] = p
                     j += n_classes
                   
            else:         
                p = run_setup(tr_scores[classifier_numbers[cl_s]], te_scores[classifier_numbers[cl_s]], X_train, y_train,
                              X_test, y_test, nmodels, n_classes, str_alg)
                stats_matrix[i, j:(j + n_classes)] = p
                j += n_classes
                 
            # print(p)
            acc_with_en, prec_with_en, rec_with_en, F1_with_en, acc_with_cls, cls_precisions_with_adj, cls_recalls_with_adj, cls_F1s_with_adj, conv_pre\
                = MATCH(X_test, y_test, te_scores, p, train_ds[0], 7, n_classes, classifiers)
            convergence_count_predict = convergence_count_predict + conv_pre
           
            
            classification_matrix[i, index_offset] = acc_with_en
            index_offset += 1

            
            for num in range(len(classifiers)):
                classification_matrix[i, index_offset] = acc_with_cls[num]
                index_offset += 1
                classification_matrix[i, index_offset:index_offset+n_classes] = cls_precisions_with_adj[num]
                index_offset += n_classes
                classification_matrix[i, index_offset:index_offset+n_classes] = cls_recalls_with_adj[num]
                index_offset += n_classes
                classification_matrix[i, index_offset:index_offset+n_classes] = cls_F1s_with_adj[num]
                index_offset += n_classes

                
        for AE_QF in range(len(algs)):
            for n in range(n_classes):
                stats_matrix[i, j] = abs(stats_matrix[i, len(stats_vec)-n_classes+n]-stats_matrix[i, len(stats_vec)+(AE_QF*n_classes)+n])
                j += 1
            stats_matrix[i, j] = sum(stats_matrix[i, j-n_classes: j])
            classification_matrix[i,index_offset]= stats_matrix[i, j]
            j += 1
            index_offset += 1
        i += 1

    # with open(dta_name+'.txt', 'a') as f:
    #     print('Convergance ratio oracle:', convergence_count_oracle, file=f)  
    #     print('Convergance ratio predict:', convergence_count_predict, file=f)
    #     print('Convergance ratio oracle:', convergence_count_oracle)  
    #     print('Convergance ratio predict:', convergence_count_predict)
    # else:
    col_names = ["Total_Samples_Used", "Training_Size", "Test_Size", "Training_Ratio", "Test_Ratio"]
    col_names += ["Training_Class_" + str(l) + "_Absolute" for l in Y]
    col_names += ["Training_Class_" + str(l) + "_Relative" for l in Y]
    col_names += ["Test_Class_" + str(l) + "_Absolute" for l in Y]
    col_names += ["Test_Class_" + str(l) + "_Relative" for l in Y]

    for alg in algs:
        for li in Y:
            col_names += [alg + "_Prediction_Class_" + str(li)]
            
    for alg in algs:
        for li in Y:
            col_names += [alg + "_AE_Class_" + str(li)]
        col_names += [alg + "_Total_AE"]

    stats_data = pd.DataFrame(data=stats_matrix,
                              columns=col_names)
    Mean_AEs = list(stats_data[[alg+'_Total_AE' for alg in algs]].mean())
    Mean_AEs.insert(0, seed)
    
    fname = res_path + dta_name
    fname += "q_details"+ "_" + strftime("%Y-%m-%d_%H-%M-%S", localtime()) + "_dir_MAE_LR " + ".csv"
    stats_data.to_csv(fname, index=False, sep=',')
    step = 'one'    
    return(Mean_AEs, step, classification_matrix)
 


def run_setup(train_scores, test_scores, X_train, y_train, X_test, y_test, nmodels, nclasses, QF):
   if QF == 'EnsembleEM':
       p = EnsembleEM(test_scores, y_train, nmodels, nclasses)
   
   elif QF == 'HDy':
       p = HDy(train_scores, test_scores, y_train)
       
   elif QF == 'CC':
       p = CC(test_scores, nclasses)     
   
   elif QF == "EDy":        
       p = EDy(train_scores, y_train, test_scores, nclasses)
       
   elif QF == 'EMQ':
       p = EMQ(test_scores, y_train, nclasses)
       
   elif QF == 'EMQ_quapy':
       p = EM_quapy(test_scores, y_train, nclasses)
  
   elif QF == 'GPAC':
        p = GPAC(train_scores, test_scores, y_train, nclasses)
        
   elif QF == 'GAC':
         p = GAC(train_scores, test_scores, y_train, nclasses)
         
   elif QF == 'FM':
         p = FM(train_scores, test_scores, y_train, nclasses)      
   
   else:
       p = eval(QF)(train_scores, test_scores, y_train, nmodels, nclasses)

   return p

if __name__ == "__main__":
    args = parse_args()
    run_synth(args.datasets, args.algorithms, args.dt, args.cl, args.mc, args.seeds)
