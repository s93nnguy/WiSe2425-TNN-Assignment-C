import numpy as np
from sklearn.cluster import KMeans


def adjust_centers_and_widths(X, k_rbf_neurons, adjust='kmean', centers=None, widths=None):
    if adjust == 'kmean':
        kmeans = KMeans(n_clusters=k_rbf_neurons)
        kmeans.fit(X)
        centers = kmeans.cluster_centers_
        widths = np.array([np.mean(np.linalg.norm(X - center, axis=1)) for center in centers])
    if adjust == 'rbf':
        pass
    return centers, widths


def load_data(file_path):
    return np.loadtxt(file_path)

def get_data_size(line):
    p, n, m = 0, 0, 0
    line_eles = line.split(' ')
    for ele in line_eles:
        if ele.startswith('P='):
            p = int(ele[ele.find('=') + 1:])
        if ele.startswith('N='):
            n = int(ele[ele.find('=') + 1:])
        if ele.startswith('M='):
            m = int(ele[ele.find('=') + 1:])
    return p, n, m

def load_file(filename):
    data = []
    header = ''
    with open(filename, 'r') as file:
        for line in file:
            # Ignore headers marked with #
            if line.startswith('#'):
                header = line
                continue
            # Split the line into input features and target values
            values = list(map(float, line.split()))
            if values == []: continue
            data.append(values)

    p, n, m = get_data_size(header)
    return p, n, m, np.array(data)

def generate_train_test_data(data, rate=0.8):
    # Calculate the number of elements for the 80% and 20% split
    num_train = int(0.8 * len(data))

    # Generate random indices for the 80% split
    train_indices = np.random.choice(len(data), num_train, replace=False)

    # Select the 80% elements
    train_data = data[train_indices]

    # Select the remaining 20% elements
    test_indices = np.setdiff1d(np.arange(len(data)), train_indices)
    test_data = data[test_indices]

    return train_data, test_data
