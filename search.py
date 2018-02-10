import torch
import math
import random

from torch.autograd import Variable
import torch.utils.data as data_utils

class Node:
    def __init__(self, parent, action, turn):
        self.parent = parent    
        self.children = []
        self.action = action
        self.turn = turn

        self.inputs = None

        self.w = 0.0
        self.n = 0
        self.s = 0.0
        self.Q = 0.0
    
    def UCT(self, logN):
        c = 1.414
        return math.inf if self.n == 0 else (self.w / self.n + c * math.sqrt(logN / self.n))
    
    def flat(self):
        yield self
        for c in self.children:
            yield from c.flat()

    def __repr__(self):
        if self.action is not None:
            return "a:%d w:%05d n:%05d s:%01d Q:%05f" % (self.action[0], self.w, self.n, self.s, self.Q)
        else:
            return "w:%05d n:%05d s:%01d Q:%05f" % (self.w, self.n, self.s, self.Q)

class MCTS:

    # self.root is the actual root of the game tree
    # self.head is the root of the subtree at the current turn
    # self.pointer is always the node of the current state (this one flies up and down the tree)

    def __init__(self, state, model):
        self.state = state
        self.model = model

        self.eps = 0
        self.gamma = 0.99

        self.reset()
    
    def reset(self):
        self.state.reset()

        self.root = Node(None, None, 1)
        self.head = self.root
        self.pointer = self.root
    
    def iterate(self):
        self._select()
        self._expand()
        self._update(self._evaluate())
        assert(self.pointer == self.head)

    def best_action(self):
        return max(self.pointer.children, key = lambda node: node.n).action
    
    def do_action(self, action):
        self.state.do_action(action)
        for node in self.head.children:
            if node.action[0] == action[0]:
                self.head = node
                self.pointer = node
                return
        node = Node(self.head, action, 1 - self.state.turn)
        self.head.children.append(node)
        self.head = node
        self.pointer = node
    
    # descends the tree to a leaf node
    def _select(self):
        if len(self.pointer.children) == 0:
            return
        # select a node
        logN = math.log(self.head.n)
        best = max(self.pointer.children, key = lambda node: node.UCT(logN))
        # update the state and move the pointer to that node, then continue descent
        self.state.do_action(best.action)
        self.pointer = best
        self._select()
    
    def _expand(self):
        _, terminal = self.state.score()
        if terminal:
            return
        if not self.pointer.n:
            self._evaluate()
        actions = self.state.actions()
        self.pointer.children = list(map(lambda a: Node(self.pointer, a, self.state.turn), actions.nonzero()))
        choice = random.choice(self.pointer.children)
        self.state.do_action(choice.action)
        self.pointer = choice

    def _rollout(self):
        score, terminal = self.state.score()
        if terminal:
            return score
        action = random.choice(self.state.actions().nonzero())
        self.state.do_action(action)
        return -self._rollout()

    def _evaluate(self):
        inputs = self.state.inputs()
        value = self.model(Variable(inputs[0][0])).data[0, 0]
        self.pointer.inputs = inputs
        self.pointer.Q = value

        self.state.save()
        rollout = self._rollout()
        self.state.restore()
        return rollout * self.eps + value * (1 - self.eps)
    
    def _update(self, score):
        self.pointer.w += score
        self.pointer.n += 1
        if self.pointer != self.head:
            self.state.undo_action(self.pointer.action)
            self.pointer = self.pointer.parent
            self._update(-score)