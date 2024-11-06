import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.Sigmoid()  # or use ReLU

    def forward(self, x):
        hidden_output = self.activation(self.fc1(x))
        output = self.activation(self.fc2(hidden_output))
        return output, hidden_output
