import random

def random_player(game):
    def player(state):
        valids = game.valid_actions(state)
        actions = [i for i in range(game.num_actions()) if valids[i]]
        return random.choice(actions)
    
    return player