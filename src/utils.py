import numpy as np
from sklearn.cluster import KMeans


def adjust_centers_and_widths(X, k_rbf_neurons):
    kmeans = KMeans(n_clusters=k_rbf_neurons)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    widths = np.array([np.mean(np.linalg.norm(X - center, axis=1)) for center in centers])
    return centers, widths


def load_data(file_path):
    return np.loadtxt(file_path)
