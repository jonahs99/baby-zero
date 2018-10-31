import numpy as np

from example.nash.nash_state import NashState, NashStatic

class NashGame:
    def __init__(self, n):
        """ Create a Game of hex
            n is the board size
        """

        self.n = n
        self.static = NashStatic(n)
    
    def num_actions(self):
        return self.n**2
    
    def num_inputs(self):
        return self.n**2 * 2

    def init_state(self):
        return NashState(self.static)

    def next_state(self, canonical_state, action):
        new_state = NashState(self.static, canonical_state)
        new_state.do_action(action)

        return new_state

    def valid_actions(self, state):
        return state.valid_actions()

    def terminal_value(self, state):
        if state.player_losing():
            return True, -1
        return False, None
    
    def str_hash(self, state):
        s = ''
        for i in range(self.n**2):
            s += str(int(state.board[0, i])) + str(int(state.board[1, i]))
        return s

    def inputs(self, state):
        return state.board.reshape((-1,))

    def display_state(self, state):
        letters = [chr(i + ord('a')) for i in range(self.n)]
        numbers = [str(i) for i in range(self.n)]
        symbols = ['.', u'\u25cb', u'\u25cf']

        X_board = state.board[0] if state.player == 1 else state.board[1]
        O_board = state.board[1] if state.player == 1 else state.board[0]
        X_board = X_board.reshape((self.n, self.n))
        O_board = O_board.reshape((self.n,self.n)).transpose()

        rep = ''
        for r in range(self.n):
            rep += ' ' * r + letters[r] + ' '
            for c in range(self.n):
                if X_board[r,c] == 1:
                    rep += symbols[1]
                elif O_board[r,c] == 1:
                    rep += symbols[2]
                else:
                    rep += symbols[0]
                rep += ' '
            rep += '\n'
        rep += ' ' * (self.n + 2) + ' '.join(numbers)

        return rep