import matplotlib.pyplot as plt
import numpy as np
import torch


def load_data(file_path, input_size, output_size):
    """Load data from a file and return input-output pairs as tensors."""
    data = np.loadtxt(file_path)

    inputs, targets = data[:, :-input_size], data[:, -output_size:]
    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)


def plot_learning_curve(losses, save_path, title):
    """Plot and save the learning curve."""
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


def generate_training_data(num_samples, n_bits, task_name):
    num_samples = min(num_samples, pow(2, n_bits))
    num_train = int(0.8 * num_samples)

    data = np.random.randint(0, 2, size=(num_samples, n_bits))

    train_indices = np.random.choice(len(data), num_train, replace=False)  # Generate random indices for the 80% split
    test_indices = np.setdiff1d(np.arange(len(data)), train_indices)  # Select the remaining 20% elements

    train_data = np.hstack((data[train_indices], data[train_indices]))
    test_data = np.hstack((data[test_indices], data[test_indices]))

    training_data = np.hstack((data, data))
    np.savetxt(f"data/encoder_decoder_{task_name}_training_data.txt", training_data, fmt='%d')

    return num_samples, train_data, test_data


def print_results(inputs, predictions, save_path):
    with open(save_path, "w") as file:
        for i, (input_pattern, prediction) in enumerate(zip(inputs, predictions)):
            text = f"Input {i + 1}: {input_pattern}, Predicted Output {i + 1}: {prediction}\n"
            print(text, end='')
            file.write(text)


def visualize_hidden_states(hidden_states, dim, save_path):
    hidden = []
    for i in range(dim):
        hidden.append(hidden_states[:, i])

    # Plotting
    fig = plt.figure(figsize=(8, 6))
    if dim == 3:
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot
        ax.scatter(hidden[0], hidden[1], hidden[2], color='b', s=50, depthshade=True)

        # Label axes
        ax.set_xlabel('Hidden Neuron 1 Activation')
        ax.set_ylabel('Hidden Neuron 2 Activation')
        ax.set_zlabel('Hidden Neuron 3 Activation')
        ax.set_title('3D Visualization of Hidden Neuron Activations')

        plt.savefig(save_path)
        # plt.show()

    if dim == 2:
        plt.scatter(hidden[0], hidden[1], color='b', marker='o', s=50)
        plt.xlabel('Hidden Neuron 1 Activation')
        plt.ylabel('Hidden Neuron 2 Activation')

        plt.title("Hidden Neuron Activations")

        plt.savefig(save_path)
        # plt.show()

    plt.close()

