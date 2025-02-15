# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 13:14:56 2022

@author: Zahra
"""

import pandas as pd
from sklearn import preprocessing
import os

def prep_data(binned=False):
    directory = os.getcwd()
    url = directory + "/data/human_activity_recognition/hars.csv"

    dta = pd.read_csv(url,
                      index_col=False,
                      skipinitialspace=True)
    #transform categorical columns to numeric
    labelencoder = preprocessing.LabelEncoder()

    objFeatures = dta.select_dtypes(include="object").columns

    for feat in objFeatures:
        dta[feat] = labelencoder.fit_transform(dta[feat].astype(str))
    
    return dta
