# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 15:31:24 2022

@author: Zahra
"""

import pandas as pd
import os

def prep_data(binned=False):
    directory = os.getcwd()
    url = directory + "/data/insects/Insects.csv"

    dta = pd.read_csv(url,
                      index_col=False,
                      skipinitialspace=True)
    # dta.rename(columns={'target': 'Class'}, inplace=True)
    
    dta.target = dta.target.replace({'aedes': 0, 'flies': 1, 'fruit': 2, 
                               'quinx': 3, 'tarsalis': 4})
    
    return dta