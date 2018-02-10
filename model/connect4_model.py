import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model_base import Model

class Connect4Model(Model):
    INPUT_SIZE = (2, 6, 7)

    def __init__(self):
        super().__init__(Net())

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
        self.conv2 = nn.Conv2d(8, 4, 2)
        self.head = nn.Linear(4 * 4 * 5, 1)

    def forward(self, x):
        x = x.view(-1, 2, 6, 7)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 4 * 4 * 5)
        x = self.head(x)
        return x