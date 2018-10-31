import numpy as np
import math
import random

EPS = 1e-8
cpuct = 8

class MCTS:
    def __init__(self, game, predictor):
        self.game = game
        self.predictor = predictor
        self.random_predictor = lambda state: (np.ones((game.num_actions(),)), 0.1*random.random())

        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s
    
    def simulate(self, state, n=100, first=False):
        for _ in range(n):
            self.search(state, first=first)
        
        s = self.game.str_hash(state)
        valids = self.Vs[s]
        Pa = [(self.Nsa[s, a] / self.Ns[s] if (s,a) in self.Nsa and valids[a] else 0)
              for a in range(self.game.num_actions())]
        Pa *= valids
        if np.sum(Pa) > 0:
            Pa /= sum(Pa)
        else:
            Pa = valids / sum(valids)
        return Pa

    def search(self, canonical_state, first=False):
        predictor = self.random_predictor if first else self.predictor

        s = self.game.str_hash(canonical_state)

        if s not in self.Es:
            terminal, value = self.game.terminal_value(canonical_state)
            self.Es[s] = (terminal, value)

        if self.Es[s][0] == True:
            # Terminal node, return the value
            return -self.Es[s][1]
        
        if s not in self.Ps:
            # Leaf node
            self.Ps[s], v = predictor(canonical_state)

            valids = self.game.valid_actions(canonical_state)
            if np.sum(valids) == 0:
                print('NO VALID MOVES')

            self.Ps[s] *= valids
            if np.sum(self.Ps[s]) > 0:
                self.Ps[s] /= np.sum(self.Ps[s])
            else:
                self.Ps[s] = valids / np.sum(valids)

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v
        
        valids = self.Vs[s]
        best_uct = -float('inf')
        best_action = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.num_actions()):
            if valids[a]:
                if (s,a) in self.Qsa:
                    uct = self.Qsa[(s,a)] + cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
                else:
                    #uct = float('inf')
                    uct = cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s] + EPS)     # Q = 0 ?
                #print('uct', (s,a), uct)
                if uct > best_uct:
                    best_uct = uct
                    best_action = a
        
        # descend via the chosen action
        next_state = self.game.next_state(canonical_state, best_action)
        v = self.search(next_state, first=first)

        # update the current node
        a = best_action

        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1
        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1
        self.Ns[s] += 1

        return -v
    
    def rollout(self, state):
        print(self.game.display_state(state))

        terminal, value = self.game.terminal_value(state)
        if terminal:
            return value

        valids = self.game.valid_actions(state)
        actions = [i for i in range(self.game.num_actions()) if valids[i]]
        next_state = self.game.next_state(state, random.choice(actions))
        return -self.rollout(next_state)