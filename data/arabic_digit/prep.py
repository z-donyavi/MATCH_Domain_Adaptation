# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 17:14:29 2022

@author: Zahra
"""

import pandas as pd
import os

def prep_data(binned=False):
    directory = os.getcwd()
    url = directory + "/data/arabic_digit/ArabicDigit.csv"

    dta = pd.read_csv(url,
                      index_col=False,
                      skipinitialspace=True)
    return dta