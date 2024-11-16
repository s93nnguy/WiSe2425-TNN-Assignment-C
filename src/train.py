import numpy as np


def train_gradient(model, X, y, learning_rate=0.01, epochs=1000):
    learning_curve = []
    for epoch in range(epochs):
        predictions = model.forward(X)
        error = y - predictions
        mse = np.mean(error ** 2)
        learning_curve.append(mse)

        rbf_activations = model.compute_rbf_activations(X)
        model.weights += learning_rate * rbf_activations.T @ error

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, MSE: {mse}")

    np.savetxt("../logs/learning_curve.txt", learning_curve)


def train_pseudo_inverse(model, X, y):
    rbf_activations = model.compute_rbf_activations(X)
    model.weights = np.linalg.pinv(rbf_activations) @ y


def train_equal_pk(model, X, y):
    rbf_activations = model.compute_rbf_activations(X)
    model.weights = np.linalg.inv(rbf_activations) @ y
