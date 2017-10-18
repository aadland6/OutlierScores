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

def complete_cases_category(df, category_col):
    """Computes the complete cases per a category
    """
    category_dict = {}
    for category in set(df[category_col]):
        print(category)
        category_df = df[df[category_col] == category]
        category_dict[category] = complete_cases(category_df)
    return category_dict

def column_completeness(df):
    """Counts the column overall column completeness
    """
    return None
# Complete cases
# Missing Data Per Column 
# Missing Data Per Column Per Category 
# Correlations between Columns
# Correlations between Columns Per Category 

if __name__ == "__main__":
    data = pd.read_csv("FakeCSV.csv")
    d = complete_cases(data)
    b = complete_cases_category(data, "FakeCategory")