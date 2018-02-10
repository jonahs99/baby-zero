from search import MCTS
import torch

class HumanAgent:
    def __init__(self, state):
        self.state = state
        self.reset()
    
    def reset(self):
        self.state.reset()
    
    def action(self):
        print('Human, please select an action:')
        actions_tens = self.state.actions().nonzero()
        actions = [a[0] for a in actions_tens]
        action = None
        while action is None:
            print(actions)
            inp = input()
            for a in actions:
                if inp == str(a):
                    action = a
        return torch.ByteTensor([action])

    def do_action(self, action):
        self.state.do_action(action)

class TrainAgent:
    def __init__(self, state, model):
        self.mcts = MCTS(state, model)
        self.reset()
    
    def reset(self):
        self.mcts.reset()
    
    def action(self, its=1000):
        for _ in range(its):
            self.mcts.iterate()
        #for node in self.mcts.head.children:
        #    print(node)
        return self.mcts.best_action()
    
    def do_action(self, action):
        self.mcts.do_action(action)