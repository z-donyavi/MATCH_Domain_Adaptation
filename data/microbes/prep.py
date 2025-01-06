# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 11:16:15 2022

@author: Zahra
"""

import pandas as pd
from sklearn import preprocessing
import os

def prep_data(binned=False):
    directory = os.getcwd()
    url = directory + "/data/microbes/microbes.csv"

    dta = pd.read_csv(url,
                      index_col=False,
                      skipinitialspace=True)
    
    values = ['Spirogyra','Ulothrix','Volvox']
    dta = dta[dta.target.isin(values) == False]
    dta = dta.reset_index(drop=True)
    
    #transform categorical columns to numeric
    labelencoder = preprocessing.LabelEncoder()

    objFeatures = dta.select_dtypes(include="object").columns

    for feat in objFeatures:
        dta[feat] = labelencoder.fit_transform(dta[feat].astype(str))
        
    
        
    return dta