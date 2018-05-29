import math
import random
import time

def current_time():
    return round(time.time() * 1000)

class MCTS:
    def __init__(self, state):
        self.state = state
        self.root = MCTSNode(None, None)
        self.pointer = self.root

        self.c = 2
    
    def think(self, max_time=1, max_its=10000):
        res = 10
        its = 0
        start_time = current_time()
        while its < max_its and current_time() - start_time < max_time * 1000:
            for _ in range(res):
                self.iterate()
            its += res
        print(its, 'iterations')

    def iterate(self):
        self._select()
        self._expand()
        score, turn = self._rollout()
        self._update(score, turn)
        assert(self.pointer == self.root)
    
    def select_action(self):
        return sorted(self.root.children, key = lambda node: node.n)[-1].action
    
    def do_action(self, action):
        self.state.make(action)
        match = list(filter(lambda node: node.action == action, self.root.children))
        if any(match):
            self.root = match[0]
        else:
            self.root = MCTSNode(self.root, action)
        self.pointer = self.root

    def _uct(self, node):
        return math.inf if node.n == 0 else node.w / node.n + self.c * math.sqrt(math.log(self.root.n) / node.n)

    def _select(self):
        if len(self.pointer.children) == 0:
            return
        best = sorted(self.pointer.children, key = lambda node: self._uct(node))[-1]
        self.state.make(best.action)
        self.pointer = best
        self._select()

    def _expand(self):
        terminal, _ = self.state.score()
        if terminal:
            return
        for action in self.state.gen_actions():
            self.pointer.add_child(action)
        self.pointer = random.choice(self.pointer.children)
        self.state.make(self.pointer.action)

    def _rollout(self):
        terminal, score = self.state.score()
        if terminal:
            return (score, self.state.turn)
        action = random.choice(self.state.gen_actions())
        self.state.make(action)
        score = self._rollout()
        self.state.unmake(action)
        return score

    def _update(self, score, turn):
        self.pointer.w += 1 - score if self.state.turn == turn else score
        self.pointer.n += 1
        if self.pointer != self.root:
            self.state.unmake(self.pointer.action)
            self.pointer = self.pointer.parent
            self._update(score, turn)

class MCTSNode:
    def __init__(self, parent, action):
        self.parent = parent
        self.children = []
        self.action = action

        self.w = 0
        self.n = 0
        
    def add_child(self, action):
        self.children.append(MCTSNode(self, action))