import numpy as np

from src.utils import adjust_centers_and_widths


class RBFNetwork:
    def __init__(self, X, n_inputs, k_rbf_neurons, m_outputs, center_adjust='kmean', centers=None, widths=None):
        self.n_inputs = n_inputs
        self.k_rbf_neurons = k_rbf_neurons
        self.m_outputs = m_outputs
        self.center_adjust = center_adjust

        # Initialize centers and widths
        self.centers, self.widths = adjust_centers_and_widths(X, k_rbf_neurons, center_adjust, centers, widths)

        # Initialize weights between -0.5 and 0.5
        np.random.seed(0)
        self.weights = np.random.uniform(-0.5, 0.5, (k_rbf_neurons, m_outputs))

    def gaussian_rbf(self, x, center, width):
        return np.exp(-np.linalg.norm(x - center) ** 2 / (2 * width ** 2))

    def compute_rbf_activations(self, X):
        rbf_activations = np.zeros((X.shape[0], self.k_rbf_neurons))
        for i, x in enumerate(X):
            for j, center in enumerate(self.centers):
                rbf_activations[i, j] = self.gaussian_rbf(x, center, self.widths[j])
        return rbf_activations

    def forward(self, X):
        rbf_activations = self.compute_rbf_activations(X)
        return rbf_activations @ self.weights
