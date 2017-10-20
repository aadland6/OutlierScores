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

def prep_data(data, keep, category_col, min_comp):
    """Calculates the completness of a dataset and imputes the median 
    """
    subset_data = data[keep]
    complete_scores = column_completeness(subset_data)
    complete_columns = [x[0] for x in complete_scores.items() if x[1] > min_comp]
    complete_subset = subset_data[complete_columns]
    complete_imputed = impute_median_category(complete_subset, category_col)
    return complete_imputed

if __name__ == "__main__":
    # data = pd.read_csv("FakeCSV.csv")
    #### diesel fuels ####
    fuel_d = pd.read_csv("pqis_d.csv")
    keep_columns_d =  list(fuel_d.columns)[0:-5]
    no_method_d = [x for x in keep_columns_d if "_M" not in x]
    clean_d = prep_data(fuel_d, no_method_d, "Fuel", .5)
    clean_d.to_csv("pqis_imputed_d.csv", index=False)
    #### aviation fuels ####
    fuel_a = pd.read_csv("pqis_a.csv")
    keep_columns_a = list(fuel_a.columns)[0:-5]
    no_method_a = [x for x in keep_columns_a if "_M" not in x]
    clean_a = prep_data(fuel_a, no_method_a, "Fuel", .5)
    clean_a.to_csv("pqis_imputed_a.csv", index=False)
