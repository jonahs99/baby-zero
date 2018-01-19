from state import State

class Connect4State(State):
    
    lines = ([ [ (col, row), (col, row+1), (col, row+2), (col, row+3) ] for row in range(3) for col in range(7) ] +
        [ [ (col, row), (col+1, row), (col+2, row), (col+3, row) ] for row in range(6) for col in range(4) ] +
        [ [ (col, row), (col+1, row+1), (col+2, row+2), (col+3, row+3) ] for row in range(3) for col in range(4) ] +
        [ [ (col, row), (col-1, row+1), (col-2, row+2), (col-3, row+3) ] for row in range(3) for col in range(3, 7) ])

    def __init__(self):
        super().__init__()
        self.n_columns = 7
        self.n_rows = 6
        self.columns = [[ 0 for _ in range(self.n_rows) ] for _ in range(self.n_columns)]
        self.turn = 1
    
    def gen_actions(self):
        return [ Connect4Action(column) for column in range(self.n_columns) if 0 in self.columns[column] ]

    def do_action(self, action):
        row = self.columns[action.column].index(0)
        self.columns[action.column][row] = self.turn
        self.turn = 3 - self.turn
    
    def undo_action(self, action):
        row = self.n_rows - 1
        while self.columns[action.column][row] == 0:
            row -= 1
        self.columns[action.column][row] = 0
        self.turn = 3 - self.turn
    
    def get_score(self):
        for line in Connect4State.lines:
            if all(self.columns[x][y] == self.turn for x, y in line):
                return 1
            if all(self.columns[x][y] == 3 - self.turn for x, y in line):
                return 0
        if any(self.columns[x][y] == 0 for x in range(self.n_columns) for y in range(self.n_rows)):
            return -1
        else:
            return 0.5

    def __repr__(self):
        symbols = ('- ', 'X ', 'O ')
        string = ''
        for row in reversed(range(self.n_rows)):
            string += ''.join(symbols[self.columns[column][row]] for column in range(self.n_columns)) + '\n'
        string += ''.join((str(column) + ' ') for column in range(self.n_columns))
        return string

class Connect4Action:
    def __init__(self, column):
        self.column = column
    def __eq__(self, other):
        return self.column == other.column
    def __repr__(self):
        return 'column ' + str(self.column)