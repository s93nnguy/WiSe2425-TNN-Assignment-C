import torch
import torch.nn as nn
import torch.optim as optim

from model import MLP


def train_model(data, input_size, hidden_size, output_size, learning_rate=0.01, epochs=1000):
    # Load data
    # inputs, targets = load_data(data_path, input_size, output_size)
    inputs, targets = torch.tensor(data[:, :-input_size], dtype=torch.float32), torch.tensor(data[:, -output_size:],
                                                                                             dtype=torch.float32)

    # Initialize model, loss function, and optimizer
    model = MLP(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs, hidden_outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # Log progress every 100 epochs
        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')

    return model, hidden_outputs, losses
