# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:07:10 2017

@author: maadland
"""
import random 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

def score_k(data, krange):
    """Generates the score and distoration for elbow plots
    """
    elbow_scores = []
    for k in krange:
        print("Evaluatng {0} clusters".format(k))
        current_model = KMeans(n_clusters=k)
        current_model.fit(data)
        current_model.score(data)
        elbow_scores.append(current_model.score(data))
    return elbow_scores 

def elbow_plot(scores, krange):
    """Generates an elbow plot for a given krange
    """
    elbow = plt.plot(krange, scores, "bx-")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Information gain")
    plt.title("Elbow plot to determine optimal K")
    return elbow
if __name__ == "__main__":
    data = pd.read_csv("FakeCSV.csv")
    data.head()
    b = score_k(data[list(data.columns)[8:]], krange=range(2,10))