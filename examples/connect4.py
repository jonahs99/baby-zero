import numpy as np
import torch

class State():
    ACTIONS = 7

    SYMMETRIES = 4
    INPUT_SHAPE = (2, 6, 7)

    lines = ([ [ (col, row), (col, row+1), (col, row+2), (col, row+3) ] for row in range(3) for col in range(7) ] +
        [ [ (col, row), (col+1, row), (col+2, row), (col+3, row) ] for row in range(6) for col in range(4) ] +
        [ [ (col, row), (col+1, row+1), (col+2, row+2), (col+3, row+3) ] for row in range(3) for col in range(4) ] +
        [ [ (col, row), (col-1, row+1), (col-2, row+2), (col-3, row+3) ] for row in range(3) for col in range(3, 7) ])
    lines_by_square = (lambda lines: [[ [l for l in lines if (col, row) in l] for row in range(6) ] for col in range(7)])(lines)

    def __init__(self):
        self.reset()
    
    def reset(self):
        self.planes = np.zeros((2, 6, 7), dtype=np.uint8) # One hot piece placements
        self.turn = 0 # 0=X, 1=O
        self.last_piece = []
        self.store = None

    # (score, terminal)
    def score(self):
        if not self.last_piece:
            return 0.0, False
        lastc, lastr = self.last_piece[-1]
        for line in State.lines_by_square[lastc][lastr]:
            if all(self.planes[1 - self.turn, row, col] == 1 for col, row in line):
                return 0.0, True
        if all(self.planes[0, 5, :] + self.planes[1, 5, :]):
            return 0.5, True
        return 0.0, False
    
    def actions(self):
        array = 1 - np.bitwise_or(self.planes[0,5,:], self.planes[1,5,:])
        return torch.from_numpy(array)

    # Returns the four symmetries
    # Flipping horizontally (1 x score) and swapping players (-1 x score)
    def inputs(self):
        symmetries = torch.FloatTensor([1, 1, -1, -1])
        inputs = np.zeros((4, 2, 6, 7), dtype=np.float32)

        inputs[0, :, :] = self.planes
        inputs[1, :, :] = np.flip(self.planes, axis=2)
        inputs[2, :, :] = np.flip(self.planes, axis=0)
        inputs[3, :, :] = np.flip(np.flip(self.planes, axis=0), axis=2)

        if self.turn == 1:
            inputs = np.flip(inputs, axis=0).copy()

        return torch.from_numpy(inputs), symmetries
    
    def do_action(self, action):
        for i in range(6):
            if self.planes[self.turn, i, action] == 0 and self.planes[1 - self.turn, i, action] == 0:
                self.planes[self.turn, i, action] = 1
                self.last_piece.append((action[0], i))
                break
        self.turn = 1 - self.turn

    def undo_action(self, action):
        self.turn = 1 - self.turn
        for i in reversed(range(6)):
            if self.planes[self.turn, i, action] == 1:
                self.planes[self.turn, i, action] = 0
                self.last_piece.pop()
                return

    def save(self):
        self.store = (self.planes.copy(), self.turn, self.last_piece.copy())
    
    def restore(self):
        self.planes, self.turn, self.last_piece = self.store

    def __repr__(self):
        symbols = ('- ', 'X ', 'O ')
        string = ''
        for row in reversed(range(6)):
            string += ''.join(symbols[self.planes[0, row, col] + 2 * self.planes[1, row, col]] for col in range(7)) + '\n'
        string += ''.join((str(col) + ' ') for col in range(7))
        return string

import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
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
    """def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 4, 4)
        self.fc1 = nn.Linear(4 * 3 * 4, 1)

    def forward(self, x):
        x = x.view(-1, 2, 6, 7)
        x = F.relu(self.conv1(x))

        x = x.view(-1, 4 * 3 * 4)
        x = self.fc1(x)

        return x"""
