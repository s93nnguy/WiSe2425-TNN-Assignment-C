import time
from model import RBFNetwork
from train import train_gradient_descent, train_pseudo_inverse
from evaluate import evaluate_model, compare_methods
from utils import load_data, adjust_centers_and_widths


def main():
    # Configuration
    n_inputs, k_rbf_neurons, m_outputs = 8, 10, 1
    learning_rate, epochs = 0.01, 1000

    # Load data
    X_train = load_data("data/training_data.txt")
    y_train = load_data("data/target_data.txt")

    # Initialize model
    model = RBFNetwork(n_inputs, k_rbf_neurons, m_outputs)
    model.centers, model.widths = adjust_centers_and_widths(X_train, k_rbf_neurons)

    # Train using gradient descent
    start_time = time.time()
    train_gradient_descent(model, X_train, y_train, learning_rate, epochs)
    computation_time_gradient = time.time() - start_time
    gradient_mse = evaluate_model(model, X_train, y_train)

    # Re-initialize model for pseudo-inverse training
    model = RBFNetwork(n_inputs, k_rbf_neurons, m_outputs)
    model.centers, model.widths = adjust_centers_and_widths(X_train, k_rbf_neurons)

    # Train using pseudo-inverse
    start_time = time.time()
    train_pseudo_inverse(model, X_train, y_train)
    computation_time_pseudo = time.time() - start_time
    pseudo_mse = evaluate_model(model, X_train, y_train)

    # Compare results
    compare_methods(gradient_mse, pseudo_mse, computation_time_gradient, computation_time_pseudo)


if __name__ == "__main__":
    main()
