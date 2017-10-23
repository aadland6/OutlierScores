# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:07:10 2017

@author: maadland
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import hdbscan
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}

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

def silhouette_k(data, n_clusters):
    """Generates a silhouette plot for n_clusters
    """
    fig, ax1 = plt.subplots(1)
    ax1.set_xlim([-.1, 1])
    ax1.set_ylim([0, data.shape[0] + (n_clusters + 1) * 10])
    clusterer = KMeans(n_clusters=n_clusters)
    cluster_labels = clusterer.fit_predict(data)
    silhouette_avg = silhouette_score(data, cluster_labels)
    silhouette_values = silhouette_samples(data, cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values, facecolor=color,
                          edgecolor=color, alpha=.7)
        ax1.text(-0.05, y_lower + .05 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    return fig, silhouette_avg

def plot_clusters(data, algorithm, args, kwds):
    """Plotting function taken from goo.gl/6vmhU5
    """
    labels = algorithm(*args, **kwds).fit_predict(data)
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    return plt


if __name__ == "__main__":
    data = pd.read_csv("FakeCSV.csv")
    data_keeps = data[list(data.columns)[8:]]
    data.head()
    b = score_k(data[list(data.columns)[8:]], krange=range(2,10))
    c = silhouette_k(data[list(data.columns)[8:]], n_clusters=5)
    plot_clusters(data_keeps, hdbscan.HDBSCAN, (), {'min_cluster_size':10})
   