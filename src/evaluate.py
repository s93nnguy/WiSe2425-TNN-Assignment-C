import numpy as np
from utils import load_data


def evaluate_model(model, X, y):
    predictions = model.forward(X)
    mse = np.mean((y - predictions) ** 2)
    return mse


def compare_methods(file_name, gradient_mse, pseudo_mse, computation_time_gradient, computation_time_pseudo):
    with open(file_name, "w") as f:
        f.write("Performance Comparison:\n")
        f.write(f"Gradient Descent MSE: {gradient_mse}\n")
        f.write(f"Pseudo-Inverse MSE: {pseudo_mse}\n")
        f.write(f"Gradient Descent Time: {computation_time_gradient} seconds\n")
        f.write(f"Pseudo-Inverse Time: {computation_time_pseudo} seconds\n")
