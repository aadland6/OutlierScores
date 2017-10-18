# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:06:50 2017

@author: maadland
"""
import pandas as pd 

def complete_cases(df):
    """Counts the compelete cases for an entire data frame 
    """
    no_na  = df.dropna(axis=0, how="any")
    return no_na.shape[0]

def complete_cases_category(df):
    """Computes the complete cases per a category
    """
    category_dict = {}
    return category_dict

# Complete cases
# Missing Data Per Column 
# Missing Data Per Column Per Category 
# Correlations between Columns
# Correlations between Columns Per Category 

if __name__ == "__main__":
    data = pd.read_csv("FakeCSV.csv")
    d = complete_cases(data)