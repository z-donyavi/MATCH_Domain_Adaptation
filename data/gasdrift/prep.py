# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 12:42:27 2022

@author: Zahra
"""

import pandas as pd
import os

def prep_data(binned=False):
    directory = os.getcwd()
    url = directory + "/data/gasdrift/gasdrift.csv"

    dta = pd.read_csv(url,
                      index_col=False,
                      skipinitialspace=True)

    
    return dta

