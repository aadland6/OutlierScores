# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 08:15:33 2017

@author: maadland
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors.lof import LocalOutlierFactor
from sklearn import svm
# from sklearn.mixture import GaussianMixture

# Constants
pd.options.mode.chained_assignment = None
RNG = np.random.RandomState(42)

def build_pca(data):
    """Creates a PCA with two dimensions for plotting the data
    """
    data_pca = PCA(n_components=2).fit_transform(data)
    data["PCA_X"] = data_pca[:, 0]
    data["PCA_Y"] = data_pca[:, 1]
    return data

def count_outliers(outlier_tuple):
    """Counts the number of tests that identified a datapoint as an outlier
    """
    outlier_count = 0
    for item in outlier_tuple:
        if item == -1:
            outlier_count += 1
    return outlier_count

def subset_df(data, target_col, columns, target):
    """Creates a training and target subset for the columns
    """
    target_subset = data[data[target_col] == target]
    training_subset = target_subset[columns]
    return target_subset, training_subset

def outlier_ensemble(data, target_col, columns):
    """Runs outlier tests for Single Class SVM, Local Outlier Factor,
       and IsolationForest for each datapoint subsetted by its target class
    """
    target_dfs = []
    for target in set(data[target_col]):
        target_subset, training_subset = subset_df(data, target_col, columns,
                                                   target)
        # initalize the classifiers
        iso_clf = IsolationForest(max_samples=100, random_state=RNG)
        lof_clf = LocalOutlierFactor(n_neighbors=20)
        svm_clf = svm.OneClassSVM(nu=.1, kernel="rbf", gamma=0.1,
                                  random_state=RNG)
        # fit the classifiers
        iso_clf.fit(training_subset)
        svm_clf.fit(training_subset)
        lof_predictions = lof_clf.fit_predict(training_subset)
        # Predict the classifers
        iso_predictions = iso_clf.predict(training_subset)
        svm_predictions = svm_clf.predict(training_subset)
        outliers_count = [count_outliers(x) for x in zip(iso_predictions,
                          lof_predictions, svm_predictions)]
        # join the data to the subset
        target_subset["IsolationPrediction"] = iso_predictions
        target_subset["LOFPrediction"] = lof_predictions
        target_subset["SVMPrediction"] = svm_predictions
        target_subset["OutlierCount"] = outliers_count
        target_dfs.append(target_subset)
    joined_df = pd.concat(target_dfs)
    return joined_df

def outliers_zscore(col, threshold=3.5):
    """Computes the modified z score for a column
    """
    median_y = np.median(col)
    mad_y = np.median([np.abs(y - median_y) for y in col])
    if mad_y > 0:
        z_scores = [.6754 * (y - median_y) / mad_y for y in col]
        outliers = [1 if z >= threshold else 0 for z in z_scores]
    else:
        print("MAD is 0")
        outliers = [0] * len(col)
    return outliers

def outliers_iqr(col):
    """Computes the IQR and returns 1 if the data is outside the bounds
    """
    quartile_1, quartile_3 = np.percentile(col, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    outliers = [1 if x > upper_bound or x < lower_bound else 0 for x in col]
    return outliers

def build_maddf(data, columns):
    """Builds a new Dataframe with the modified Z scores on each column
    """
    column_dict = {}
    for column in columns:
        column_key = "{0}_MAD".format(column)
        column_dict[column_key] = outliers_zscore(data[column])
    return column_dict

def global_mad(data, target_col, columns):
    """Calculates the outliers for the whole dataframe
    """
    df_list = []
    for target in set(data[target_col]):
        target_subset, training_subset = subset_df(data, target_col, columns,
                                                   target)
        mad_df = pd.DataFrame.from_dict(build_maddf(training_subset, columns))
        df_list.append(mad_df)
    outlier_columns = pd.concat(df_list)
    return outlier_columns

def row_outliers(row):
    """Appends the column name where an outlier was detected
    """
    outlier_cols = []
    for item in row.index:
        if row[item] == 1:
            outlier_cols.append(item)
    return " ".join(outlier_cols)


if __name__ == "__main__":
    fuel_data = pd.read_csv("fuel_d_imputed.csv")
    keep_columns = ['Fuel', 'Acid  No', 'Sulfur', 'Tr-Ca', 
                    'Tr-Pb', 'Tr-Na+K', 'Tr-V', '90% Rec', 'FBP', 
                    '%R&L', 'Flash', 'Density',
                    'Viscosity', 'Cloud Pt', 'Pour Pt', 
                    'Ash', 'Carb Res', 'Acc Stab', 'PC', 'Demuls', 'Cet Indx']
    fuel_trimmed = fuel_data[keep_columns].dropna(axis=0, how="any")
    sans_fuel =[x for x in keep_columns if x != "Fuel"]
    fuel_pre_pca = fuel_trimmed[sans_fuel]
    fuel_post_pca = build_pca(fuel_pre_pca)
    fuel_post_pca["Fuel"] = fuel_trimmed["Fuel"]
    fuel_post_pca.to_csv("FuelsPostPCA.csv", index=False)
    