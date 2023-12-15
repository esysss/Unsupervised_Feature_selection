from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

def kmeans(x,k):
    means = KMeans(n_clusters=k, init='random').fit(x)
    return means.labels_

def spectoral(x,k):
    clustering = SpectralClustering(n_clusters= k, assign_labels = 'discretize').fit(x)
    return clustering.labels_
