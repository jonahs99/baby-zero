from state import State

size = 5

class NashState(State):

    # a set for each hex's neighbors
    # note: this is a board shaped like \\ not //
    connections = [
        { (nr*size + nc) for nr,nc in {(r,c+1),(r-1,c+1),(r-1,c),(r,c-1),(r+1,c-1),(r+1,c)} if 0 <= nr < size and 0 <= nc < size }
        for r in range(size) for c in range(size) ]

    left_edge = { (r * size) for r in range(size) }
    right_edge = { (r * size + size - 1) for r in range(size) }
    top_edge = { (c) for c in range(size) }
    bottom_edge = { (c + size * (size - 1)) for c in range(size) }

    for i in left_edge:
        connections[i].add(-1)
    for i in right_edge:
        connections[i].add(-2)
    for i in top_edge:
        connections[i].add(-3)
    for i in bottom_edge:
        connections[i].add(-4)

    symbols = ['.', 'X', 'O']
    letters = [ chr(i + ord('a')) for i in range(26) ]
    numbers = [ str(i) for i in range(26) ]

    def __init__(self):
        super().__init__()
        self.size = size
        self.board = [0 for _ in range(self.size * self.size)]
        self.turn = 1

        self.nodes = [[], []]
    
    def gen_actions(self):
        return [ NashAction(i) for i in range(self.size ** 2) if self.board[i] == 0 ]
    
    def do_action(self, action):
        player_list = self.nodes[self.turn - 1]
        action.nodes = player_list.copy()

        connected = [node for node in player_list if action.index in node]
        if len(connected):
            action_node = set.union(self.connections[action.index], *connected)
            self.nodes[self.turn - 1] = [ node for node in player_list if node not in connected ] + [action_node]
        else:
            action_node = self.connections[action.index].copy()
            self.nodes[self.turn - 1] += [action_node]

        self.board[action.index] = self.turn
        self.turn = 3 - self.turn
    
    def undo_action(self, action):
        self.board[action.index] = 0
        self.turn = 3 - self.turn

        self.nodes[self.turn - 1] = action.nodes
    
    def get_score(self):
        if any((-1 in node and -2 in node) for node in self.nodes[0]):
            return 1 if self.turn == 1 else 0
        if any((-3 in node and -4 in node) for node in self.nodes[1]):
            return 1 if self.turn == 2 else 0
        return -1
    
    def __repr__(self):
        ret = ''
        for r in range(self.size):
            ret += ' ' * r + self.letters[r] + ' '
            for c in range(self.size):
                ret += self.symbols[self.board[r * self.size + c]] + ' '
            ret += '\n'
        ret += ' ' * (self.size + 2) + ''.join([self.numbers[i] + ' ' for i in range(self.size)])
        return ret

class NashAction:
    letters = [ chr(ord('a') + i) for i in range(26) ]
    numbers = [ str(i) for i in range(26) ]

    def __init__(self, index):
        self.index = index
        self.nodes = None

    def __eq__(self, other):
        return self.index == other.index

    def __repr__(self):
        return NashAction.letters[self.index // NashState.size] + NashAction.numbers[self.index % NashState.size]
