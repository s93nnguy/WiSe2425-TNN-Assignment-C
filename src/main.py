import time
import copy
from model import RBFNetwork
from train import train_gradient, train_pseudo_inverse, train_equal_pk
from evaluate import evaluate_model, compare_methods
from utils import *


def rbf_gradient(model, X_train, y_train, X_test, y_test):
    # Train using gradient descent
    start_time = time.time()
    train_gradient(model, X_train, y_train)
    running_time = time.time() - start_time

    mse = evaluate_model(model, X_test, y_test)

    return mse, running_time


def rbf_pseudo_inverse(model, X_train, y_train, X_test, y_test):
    # Train using pseudo inverse
    start_time = time.time()
    train_pseudo_inverse(model, X_train, y_train)
    running_time = time.time() - start_time

    mse = evaluate_model(model, X_test, y_test)

    return mse, running_time


def main():
    # Configuration
    k, learning_rate, epochs = 2, 0.01, 1000

    # Load data
    p, n, m, data = load_file('../data/PA-A_training_data_06.txt')
    training_data, test_data = generate_train_test_data(data, rate=0.8)
    X_train, y_train = training_data[:, :-m], training_data[:, -m:]
    X_test, y_test = training_data[:, :-m], training_data[:, -m:]

    p = X_train.shape[0]

    if p == k:
        model = RBFNetwork(X_train, n, k, m)
        train_equal_pk(model, X_train, y_train)
        mse = evaluate_model(model, X_test, y_test)

        print(mse)
    elif p < k:
        raise Exception(f"{p} < {k}")
    else:
        # Initialize model
        model_gradient = RBFNetwork(X_train, n, k, m)
        model_pseudo = copy.deepcopy(model_gradient)

        # Train using gradient descent
        gradient_mse, computation_time_gradient = rbf_gradient(model_gradient, X_train, y_train, X_test, y_test)

        # Train using pseudo-inverse
        pseudo_mse, computation_time_pseudo = rbf_pseudo_inverse(model_pseudo, X_train, y_train, X_test, y_test)

        # Compare results
        compare_methods(gradient_mse, pseudo_mse, computation_time_gradient, computation_time_pseudo)


if __name__ == "__main__":
    main()
