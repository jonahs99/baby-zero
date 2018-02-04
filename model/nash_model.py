import torch
import torch.nn as nn
import torch.optim as optim

import nash

class Model:
    INPUT_SIZE = (2, nash.size, nash.size)

    map1 = [(i, i % nash.size, i // nash.size) for i in range(nash.size ** 2)]
    map2 = [(i, i // nash.size, i % nash.size) for i in range(nash.size ** 2)]

    def __init__(self):
        self.net = Net()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9)
        self.criterion = nn.MSELoss()

    def represent(self, state):
        # we transpose the board if it is turn 2

        planes = torch.zeros(*Model.INPUT_SIZE)
        for i, col, row in (Model.map1 if state.turn == 1 else Model.map2):
            if state.board[i] == state.turn:
                planes[0, col, row] = 1
            elif state.board[i] == 3 - state.turn:
                planes[1, col, row] = 1
        return planes

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 8, 3)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(8, 4, 2)
        self.relu2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(4, 4, 2)
        self.relu3 = nn.LeakyReLU()
        self.fc1 = nn.Linear(4 * (nash.size - 4) ** 2, 1)

    def forward(self, x):
        x = x.view(-1, *Model.INPUT_SIZE)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(-1, 4 * (nash.size - 4) ** 2)
        x = self.fc1(x)
        return x