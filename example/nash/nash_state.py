import numpy as np

import copy

class NashStatic:
    def __init__(self, n):
        self.n = n
        self.adj = self._adjacencies()
    
    def _adjacencies(self):
        def index(r, c):
            return r * self.n + c
        def adjacent_to(r, c):
            return [(r,c+1),(r-1,c+1),(r-1,c),(r,c-1),(r+1,c-1),(r+1,c)]
        def in_bounds(r, c):
            return 0 <= r < self.n and 0 <= c < self.n
        adj = [{index(ar, ac) for ar,ac in adjacent_to(r, c) if in_bounds(ar, ac)}
               for r in range(self.n) for c in range(self.n) ]

        for r in range(self.n):
            adj[index(r, 0)].add(-1)
            adj[index(r, self.n-1)].add(-2)

        return [frozenset(a) for a in adj]

class NashState:
    def __init__(self, static, state=None):
        self.static = static
        self.n = static.n
        if state is None:
            self.board = np.zeros((2, static.n**2), dtype=float)
            self.nodes = [frozenset(), frozenset()]
            self.player = 1
        else:
            self.board = state.board.copy()
            self.nodes = state.nodes.copy()
            self.player = state.player

    def valid_actions(self):
        # Returns 1 where board[0] and Transpose(board[1]) are empty
        empty_0 = 1 - self.board[0]
        empty_1 = 1 - (self.board[1].reshape((self.n, self.n)).transpose().reshape((-1,)))
        return empty_0 * empty_1

    def do_action(self, action):
        # Place the piece on the board
        assert(self.board[0, action] == 0)
        self.board[0, action] = 1

        # Update nodes
        connected_nodes = {n for n in self.nodes[0] if action in n}
        if len(connected_nodes):
            union_node = frozenset.union(self.static.adj[action], *connected_nodes)
            self.nodes[0] = self.nodes[0] - connected_nodes | {union_node}
        else:
            node = self.static.adj[action]
            self.nodes[0] |= {node}
        
        # Player switches
        self._flip()
    
    def player_losing(self):
        # Did the opponent make a connection?
        return any(-1 in node and -2 in node for node in self.nodes[1])
    
    def _flip(self):
        # Switches the POV of the board
        self.player = -self.player
        self.board = np.flip(self.board, 0)
        self.nodes[0], self.nodes[1] = self.nodes[1], self.nodes[0]