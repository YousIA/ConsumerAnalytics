from itertools import combinations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import warnings
import sqlite3
import umap
import seaborn as sns
import networkx as nx
import time
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import BisectingKMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from yellowbrick.cluster import KElbowVisualizer
import scipy.cluster.hierarchy as shc
from scipy.spatial.distance import cdist
from sklearn import mixture
from gap_statistic import OptimalK
from sklearn.manifold import trustworthiness
#from spmf import Spmf
#import tensorflow as ts
#from keras.models import Sequential
#from keras.layers import Dense

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
pd.options.mode.chained_assignment = None  # default='warn'


def qty_support(df):
    # Create an empty dictionary to store the support values for each item and quantity
    support_dict = {}

    # Iterate through each item in the dataframe
    for item in df.columns:
        # Create an empty dictionary to store the support values for each quantity of the item
        support_by_quantity = {}

        # Iterate through each quantity of the item (from 1 to the maximum quantity)
        for count in range(1, int(df[item].max()) + 1):
            # Calculate the support for the current quantity of the item (ignoring transactions with quantity 0)
            count_support = len(df[(df[item] == count)]) / len(df[df[item] > 0])

            # Add the support value to the dictionary of supports for the current item
            support_by_quantity[count] = count_support

        # Add the dictionary of supports for the current item to the overall dictionary of supports
        support_dict[item] = support_by_quantity

    # Convert the dictionary of supports to a dataframe and format the index and column labels
    df_support = pd.DataFrame.from_dict(support_dict, orient='index')
    df_support.index.name = 'Item'
    df_support.columns.name = 'Quantity'

    # Return the dataframe of support values
    return df_support

def gmm_algo(df, n_components=4):
    gmm = mixture.GaussianMixture(n_components).fit(df)
    labels = gmm.predict(df)
    plt.scatter(df[:, 0], df[:, 1], c=labels, s=40, cmap='viridis')

    tmp_reduced_data_df = pd.DataFrame(df, index=df.index)
    tmp_reduced_data_df['cluster'] = labels

    return tmp_reduced_data_df

def kmeans_overlap(df, n_clusters=4, ax=None):
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(df)

    # plot the input data
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(df[:, 0], df[:, 1], c=labels, s=40, cmap='viridis', zorder=2)

    # plot the representation of the KMeans model
    centers = kmeans.cluster_centers_
    radii = [cdist(df[labels == i], [center]).max()
             for i, center in enumerate(centers)]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))

def analyze_cluster(df):
    #Silhoutte score and elbow-method
    model = KMeans()
    # k is range of number of clusters.
    v = KElbowVisualizer(model, k=(2, 10), metric='silhouette', timings=True)
    v.fit(df)  # Fit the data to the visualizer
    v.show()

    # k is range of number of clusters.
    v = KElbowVisualizer(model, k=(2, 10), metric='calinski_harabasz', timings=True)
    v.fit(df)  # Fit the data to the visualizer
    v.show()  # Finalize and render the figure

    # Dendogram for Heirarchical Clustering
    #plt.figure(figsize=(10, 7))
    #plt.title("Dendrograms")
    #shc.dendrogram(shc.linkage(df, method='ward'))

def ch_review(df):
    model = KMeans()
    # k is range of number of clusters.
    v = KElbowVisualizer(model, k=(2, 10), metric='calinski_harabasz', timings=True)
    v.fit(df)  # Fit the data to the visualizer
    v.show()  # Finalize and render the figure

# Datahandling
def new_df(fileName):
    # Read file
    df = pd.read_excel(fileName)

    # Postgres-standard
    df['date'] = pd.to_datetime(df['date'])
    df['date'].dt.strftime('%Y-%m-%d')

    # Create a unique transaction id using store number and date
    df['tnr'] = df['store'].astype(str) + df['date'].astype(str) + df['tnr'].astype(str)
    df['tnr'].replace(to_replace='-', value='', regex=True, inplace=True)

    # Fill NaN with 0, set tnr as index and drop store and date column
    df.fillna(0, inplace=True)
    df.set_index('tnr', inplace=True)
    df.drop(['store', 'date'], axis=1, inplace=True)

    return df

#https://github.com/milesgranger/gap_statistic/blob/master/README.md
def gap_stat(df):
    optK = OptimalK(parallel_backend='rust')
    range_n_clusters = range(2, 9)
    n_clusters = optK(df, cluster_array=range_n_clusters)
    print('Optimal clusters: ', n_clusters)
    OptimalK.plot_results(optK)
    return(optK.gap_df)

def db_score(df):
    range_n_clusters = range(2, 9)
    for n in range_n_clusters:
        kmeans = KMeans(n_clusters=n, random_state=42).fit(df)
        # we store the cluster labels
        labels = kmeans.labels_
        score = davies_bouldin_score(df, labels)
        print(f"For n_clusters = {n}, the DB-score is : {score:.4f}")

# Takes grouped and padded vgr df
def dbscan_algo(X):
    db = DBSCAN(eps=1, min_samples=50).fit(X)
    labels = db.labels_

    # Get the number of unique labels
    num_labels = len(np.unique(labels))
    print(num_labels)

    # Create a scatter plot of the points, coloring each point according to its label
    for i, label in enumerate(labels):
        plt.scatter(X[i, 0], X[i, 1])

    plt.show()


# Takes a df where each row needs the same number of columns, and the nr of clusters
def kmeans_algo(reduced_data, n_clusters, df):
    st = time.time()

    # Fit the KMeans model to the reduced data
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(reduced_data)

    # Get the cluster labels for each row
    labels = kmeans.labels_

    # Plot the clusters
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'w']
    plt.figure()
    for i in range(n_clusters):
        plt.scatter(reduced_data[labels == i, 0], reduced_data[labels == i, 1], c=colors[i], label='Cluster ' + str(i))
    plt.legend()

    # Plot the centroids
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=3, color='k')

    plt.show()

    # Add the cluster labels as a new column in the reduced data
    tmp_reduced_data_df = pd.DataFrame(reduced_data, index=df.index)
    tmp_reduced_data_df['cluster'] = labels

    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

    return tmp_reduced_data_df

def bkmeans_algo(reduced_data, n_clusters, df):
    st = time.time()

    # Fit the KMeans model to the reduced data
    bkmeans = BisectingKMeans(n_clusters=n_clusters, random_state=42)
    bkmeans.fit(reduced_data)

    # Get the cluster labels for each row
    labels = bkmeans.labels_

    # Plot the clusters
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
    plt.figure()
    for i in range(n_clusters):
        plt.scatter(reduced_data[labels == i, 0], reduced_data[labels == i, 1], c=colors[i], label='Cluster ' + str(i))
    plt.legend()

    # Plot the centroids
    centroids = bkmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=3, color='k')

    plt.show()

    # Add the cluster labels as a new column in the reduced data
    tmp_reduced_data_df = pd.DataFrame(reduced_data, index=df.index)
    tmp_reduced_data_df['cluster'] = labels

    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

    return tmp_reduced_data_df

def rules_to_excel(df, n_clusters, method):
    i = 0
    while i < n_clusters:
        tmp_df = df[df['cluster'] == i]
        tmp_df.drop(['cluster'], axis=1, inplace=True)
        tmp_rules = apriori_algo(tmp_df)
        tmp_rules.to_excel(f"{method}_cluster{i}.xlsx")
        i += 1


    # Takes grouped_vgr df
def apriori_algo(df, min_support=0.05):
    st = time.time()

    basket_sets = df.applymap(encode_units)

    frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)

    frequent_itemsets.sort_values('support', ascending=False, inplace=True)
    top_frequent_combinations = frequent_itemsets.head(10)

    print(top_frequent_combinations)

    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.20)
    rules.sort_values('confidence', ascending=False, inplace=True)

    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

    return rules


def fpgrowth_algo(df):
    st = time.time()
    basket_sets = df.applymap(encode_units)

    # Perform FP-growth
    frequent_itemsets = fpgrowth(basket_sets, min_support=0.1, use_colnames=True)

    # Print the frequent itemsets
    print(frequent_itemsets)

    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')


def encode_units(x):
    if x <= 0:
        return 0
    if x > 0:
        return 1


def elbow_method(df):
    Sum_of_squared_distances = []
    K = range(1, 10)
    for num_clusters in K:
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(df)
        Sum_of_squared_distances.append(kmeans.inertia_)

    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Squared distances')
    plt.show()


# Methods for component reduction

#PCA-reduction
def pca_reduction(df, n_components):
    st = time.time()

    # Fit the PCA model to the data
    pca = PCA(n_components=n_components)
    pca.fit(df)

    # Transform the data to the new PCA-reduced space
    reduced_components = pca.transform(df)

    # Convert the reduced components to a dataframe with the same index
    #reduced_df = pd.DataFrame(reduced_components, index=df.index)

    et = time.time()
    elapsed_time = et - st
    print("%.2f" % elapsed_time)

    return reduced_components

#UMAP-reduction
def umap_reduction(df, n_components):
    st = time.time()

    # Returns the array with the reduced components. Random state seed to ensure same result each time
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    embedding = reducer.fit_transform(df)

    et = time.time()
    elapsed_time = et - st
    print("%.2f" % elapsed_time)

    return embedding

#Calculate score for non-linear reduction techniques
def score_nl_rt(df, reduced_df, n_neighbors=2):
    st = time.time()
    trust_score = trustworthiness(df, reduced_df, n_neighbors=n_neighbors)
    continuity_score = trustworthiness(reduced_df, df, n_neighbors=n_neighbors)

    print(f"Trustworthiness score: {trust_score}")
    print(f"Continuity score: {continuity_score}")

    et = time.time()
    elapsed_time = et - st
    print("%.2f" % elapsed_time)

#Variance for the linear reduction techniques
def variance_l_rt(df, method, n_components=10):
    if method == 'pca':
        pca = PCA(n_components=n_components, random_state=6)
        pca.fit(df)
        print("Explained Variance Ratio:", pca.explained_variance_ratio_)
    elif method == 'svd':
        svd = TruncatedSVD(n_components=n_components, random_state=6)
        svd.fit(df)
        print("Explained Variance Ratio:", svd.explained_variance_ratio_)


#UMAP-reduction with variance. Extremely slow.
def umap_reduction_variance(df, n_components):
    st = time.time()
    # Returns the dataframe with the reduced components
    reducer = umap.UMAP(n_components=n_components)
    reduced_data = reducer.fit_transform(df)
    df_embedding = pd.DataFrame(reduced_data, index=df.index)

    # Explains the variance
    reconstruction = reducer.inverse_transform(reduced_data)
    reconstruction_error = np.sum((df - reconstruction) ** 2)
    total_variance = np.sum((df - np.mean(df, axis=0)) ** 2)
    explained_variance = 1.0 - reconstruction_error / total_variance

    et = time.time()
    elapsed_time = et - st
    print("%.2f" % elapsed_time)

    return df_embedding, explained_variance

#t-SNE reduction
def tsne_reduction(df, n_components, n=30.0):
    st = time.time()

    reducer = TSNE(n_components=n_components, perplexity=n, random_state=42)
    reduced_data = reducer.fit_transform(df)

    et = time.time()
    elapsed_time = et - st
    print("%.2f" % elapsed_time)

    return reduced_data

def svd_reduction(df, n_components):
    st = time.time()

    svd = TruncatedSVD(n_components=n_components)
    svd.fit(df)
    svd_transform = svd.transform(df)
    #svd_df = pd.DataFrame(svd_transform)
    #svd_df.index = df.index

    et = time.time()
    elapsed_time = et - st
    print("%.2f" % elapsed_time)

    return svd_transform

def reduce_comp(df):
    # Reduce to number of components
    num_components = 2
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    print(X_scaled)
    # Create a PCA model to reduce our data to 2 dimensions for visualisation
    pca = PCA(n_components=num_components)
    pca.fit(X_scaled)

    # Transfor the scaled data to the new PCA space
    X_reduced = pca.transform(X_scaled)

    # Convert to a data frame
    X_reduceddf = pd.DataFrame(X_reduced, columns=['PC1', 'PC2'])
    # X_reduceddf = pd.DataFrame(X_reduced, columns=['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8'])

    return X_reduceddf


# Count number of items in a cluster
def count_items_in_cluster(df, cluster_num):
    # Filter the dataframe to only include transactions in the specified cluster
    df_cluster = df[df['cluster'] == cluster_num]

    # Create a dictionary to store the count of each item
    item_counts = {}

    # Iterate through each item in the dataframe
    for item in df.columns[:-1]:
        # Count the number of transactions in the cluster that contain the item
        count = len(df_cluster[df_cluster[item] != 0])
        # Add the item and its count to the dictionary
        item_counts[item] = count

    # Convert the dictionary to a dataframe and sort it by the count in descending order
    item_counts_df = pd.DataFrame.from_dict(item_counts, orient='index', columns=['count'])
    item_counts_df.sort_values(by='count', ascending=False, inplace=True)

    return item_counts_df


# Total number of transactions in cluster
def count_rows_in_cluster(df, cluster):
    # select only the rows in the dataframe where the cluster column matches the desired cluster
    cluster_df = df.loc[df['cluster'] == cluster]

    # get the number of rows
    num_rows = cluster_df.shape[0]
    print(num_rows)

    return num_rows


# Apriori and association graph
def associationg_graph(df):
    # Generate association rules using Apriori
    rules = apriori_algo(df)
    # Create the graph
    G = nx.Graph()
    # Add nodes for each unique antecedent and consequent
    for rule in rules:
        antecedent = list(rule[0])[0]
        consequent = list(rule[1])[0]
        G.add_node(antecedent)
        G.add_node(consequent)

    # Connect the antecedent node to the consequent node
    for rule in rules:
        antecedent = list(rule[0])[0]
        consequent = list(rule[1])[0]
        G.add_edge(antecedent, consequent)

    # Visualize the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels)
    plt.show()


def eclat(df, min_support):
    def get_support(df, items):
        rows = np.ones(df.shape[0], dtype=bool)
        for item in items:
            rows = np.logical_and(rows, df[item] == 1)
        return np.sum(rows)

    freq_items = []
    items = list(df.columns)
    for k in range(1, len(items) + 1):
        for itemset in combinations(items, k):
            support = get_support(df, itemset) / len(df)
            if support >= min_support:
                freq_items.append(set(itemset))

    return freq_items


def relim(df, min_support):
    def get_support(df, items):
        rows = np.ones(df.shape[0], dtype=bool)
        for item in items:
            rows = np.logical_and(rows, df[item] == 1)
        return np.sum(rows)

    freq_items = []
    items = list(df.columns)
    for k in range(1, len(items) + 1):
        for itemset in combinations(items, k):
            support = get_support(df, itemset) / len(df)
            if support >= min_support:
                freq_items.append(set(itemset))
                df_new = df.loc[:, list(itemset)]
                support_dict = df_new.all(axis=1).value_counts(normalize=True).to_dict()
                for key, value in support_dict.items():
                    if key:
                        support_new = value * support
                        if support_new >= min_support:
                            freq_items.append(set(itemset))

    return freq_items

#Takes merged df that contains cluster information
def top10_cluster(df, n_clusters):
    n = 0
    while n < n_clusters:
        print("\n" + "Cluster %s" % n)
        print(count_items_in_cluster(df, n).head(10))
        count_rows_in_cluster(df, n)
        n += 1

def drop_df_cluster(df):
    df.drop(['cluster'], axis=1, inplace=True)
    return df

#Takes merged df
def arm_cluster(df, n_clusters):
    n = 0
    while n < n_clusters:
        print("\n" + "Cluster %s" % n)
        tmp_df = df[df['cluster'] == n]
        drop_df_cluster(tmp_df)
        tmp_var = apriori_algo(tmp_df)
        print(tmp_var)
        #fpgrowth_algo(tmp_df)
        n += 1

def silhouette_analysis(df):
    range_n_clusters = range(2, 9)

    for n_clusters in range_n_clusters:
        # Initialize the KMeans algorithm with n_clusters
        kmeans = KMeans(n_clusters=n_clusters)
        cluster_labels = kmeans.fit_predict(df)

        # Calculate silhouette score for each cluster number
        silhouette_avg = silhouette_score(df, cluster_labels)

        print(f"For n_clusters = {n_clusters}, the average silhouette_score is : {silhouette_avg:.4f}")