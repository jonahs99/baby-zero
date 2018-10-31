from zero import MCTS
from numpy.random import choice

class Trainer:
    def __init__(self, game, net):
        self.game = game
        self.net = net
    
    def train(self, epochs=10, games=10, gens=10, sims=100, callback=lambda: None):
        for i_epoch in range(epochs):
            train_examples = []
            for _ in range(games):
                #print('*********')
                train_examples += self.self_play(sims=sims, no_net=(self.net.winner_genome is None))
            self.net.train(train_examples, generations=gens)
            callback(i_epoch)

    def self_play(self, sims=100, no_net=False):
        # Plays a self play game and returns training examples

        train_examples = []

        mcts = MCTS(self.game, self.net.predictor())
        state = self.game.init_state()
        game_value = None

        while True:
            #print(self.game.display_state(state))

            terminal, value = self.game.terminal_value(state)
            if terminal:
                game_value = value * state.player
                break
            Psa = mcts.simulate(state, first=no_net, n=sims)

            train_examples.append([self.game.inputs(state), Psa, state.player])

            action = choice(self.game.num_actions(), size=None, p=Psa)
            state = self.game.next_state(state, action)
        
        for ex in train_examples:
            ex[2] *= game_value
        
        return train_examples
