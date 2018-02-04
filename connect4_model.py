import torch
import torch.nn as nn
import torch.optim as optim

class Model:
    INPUT_SIZE = (2, 6, 7)

    def __init__(self):
        self.net = Net()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.004, momentum=0.9)
        self.criterion = nn.MSELoss()

    def represent(self, state):
        planes = torch.zeros(2, 6, 7)
        for i, column in enumerate(state.columns):
            for j, value in enumerate(column):
                if value == state.turn:
                    planes[0, j, i] = 1.0
                elif value == 3 - state.turn:
                    planes[1, j, i] = 1.0
        return planes

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 8, 2)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(8, 4, 2)
        self.relu2 = nn.LeakyReLU()
        self.fc1 = nn.Linear(4 * 4 * 5, 1)

    def forward(self, x):
        x = x.view(-1, 2, 6, 7)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.view(-1, 4 * 4 * 5)
        x = self.fc1(x)
        return x