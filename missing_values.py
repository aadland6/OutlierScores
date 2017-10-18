# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:06:50 2017

@author: maadland
"""
import pandas as pd 
import numpy as np

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
    column_dict = {}
    for column in df.columns:
        column_dict[column] =  len(df[column].dropna()) / len(df[column])
    return column_dict

def column_completeness_category(df, category_col):
    """Computes the column completeness by category
    """
    category_column_dict = {}
    for category in set(df[category_col]):
        category_df = df[df[category_col] == category]
        category_column_dict[category] = column_completeness(category_df)
    return category_column_dict

def column_median_category(df, category_col):
    """Computes the median value for the category
    """
    category_median_dict = {}
    for category in set(df[category_col]):
        category_df = df[df[category_col] == category]
        category_medians = {}
        for column in category_df.columns:
            try:
                category_medians[column] = np.nanmedian(category_df[column])
            except TypeError:
                category_medians[column] = ""
        category_median_dict[category] = category_medians
    return category_median_dict

def impute_median_category(df, category_col):
    """Imputes the median of the data by substted by the category column
    """
    median_dfs = []
    for category in set(df[category_col]):
        category_df = df[df[category_col] == category]
        median_dfs.append(category_df.fillna(category_df.median()))
    return pd.concat(median_dfs)        

if __name__ == "__main__":
    data = pd.read_csv("FakeCSV.csv")
    #### fuels ####
    fuel_d = pd.read_csv("pqis_d.csv")
    keep_columns =  list(fuel_d.columns)[0:-5]
    no_method = [x for x in keep_columns if "_M" not in x]
    fuel_d_clean = fuel_d[no_method] 
    fuel_column_scores = column_completeness(fuel_d_clean)
    fuel_column_subset = [x[0] for x in fuel_column_scores.items() if x[1] > .69]
    fuel_d_clean_column_subset = fuel_d_clean[fuel_column_subset]
    fuel_imputed = impute_median_category(fuel_d_clean_column_subset, "Fuel")
    fuel_imputed.to_csv("fuel_d_imputed.csv", index=False)
    
    