import torch


def evaluate_model(model, data, input_size, output_size):
    inputs, targets = torch.tensor(data[:, :-input_size], dtype=torch.float32), torch.tensor(data[:, -output_size:],
                                                                                             dtype=torch.float32)
    model.eval()

    with torch.no_grad():
        predictions, hidden_outputs = model(inputs)
    mse = torch.mean((predictions - targets) ** 2).item()

    return predictions, hidden_outputs, mse
