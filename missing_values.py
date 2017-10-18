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
# Complete cases
# Missing Data Per Column 
# Missing Data Per Column Per Category 
# Correlations between Columns
# Correlations between Columns Per Category 

if __name__ == "__main__":
    data = pd.read_csv("FakeCSV.csv")
    a = complete_cases(data)
    b = complete_cases_category(data, "FakeCategory")
    c = column_completeness(data)
    d = column_completeness_category(data, "FakeCategory")
    #### fuels ####
    fuel_d = pd.read_csv("pqis_d.csv")
    keep_columns =  list(fuel_d.columns)[0:-5]
    fuel_d_clean = fuel_d[keep_columns] 
    af = complete_cases(fuel_d_clean)
    bf = column_completeness(fuel_d_clean)
    out_bf = pd.DataFrame({"ColumnName":list(bf.keys()),
                           "Completeness":list(bf.values())})
    out_bf.to_csv("FuelCompleteness.csv", index=False)
    