# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 14:33:29 2022

@author: Zahra
"""

import pandas as pd
import os

def prep_data(binned=False):
    directory = os.getcwd()
    url = directory + "/data/statlog/statlog.csv"

    dta = pd.read_csv(url,
                      index_col=False,
                      skipinitialspace=True)

    
    return dta