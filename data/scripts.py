# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 14:55:13 2023

@author: z5343492
"""
import pandas as pd
import io



data_set_index = pd.read_csv("data_index.csv",
                             sep=";",
                             index_col="dataset")

df_ind = data_set_index.loc[data_set_index["classes"] > 2]
data_sets = list(df_ind.index)
abbrev = list(data_set_index["abbr"].loc[df_ind.index])

for i in range(len(data_sets)):
    f = io.open("cl_"+ str(abbrev[i]) + ".txt", 'w', newline='\n')

    f.writelines("#!/bin/bash\n") 
    f.writelines("#SBATCH -t 5:00:00 -c 16\n")
    f.writelines("#SBATCH --mail-user=z.donyavi@unsw.edu.au --mail-type=END,FAIL\n")
    f.writelines('export INPUT="*"\n')
    f.writelines('export OUTPUT="*"\n')
    f.writelines('export WAIT_CHECKPOINT="3600"\n')
    f.writelines("module load anaconda3\n")
    f.writelines("source activate Python3.9.11\n")
    f.writelines("cd /home/zdonyavi/Ensemble_Quantifiers_MC_SQ_new_datasets\n")
    f.writelines("python3 Classification_pablo.py -d " + str(data_sets[i] + " -a MCMQ EDy EMQ"))
    # f.writelines("python3 run_forest.py -d " + str(data_sets[i] + ' -a readme ED CC PWK  EM GAC GPAC FM'))
    f.close()