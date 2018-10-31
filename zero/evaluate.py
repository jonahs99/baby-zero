class Arena:
    def __init__(self, game, player1, player2):
        self.game = game
        self.players = {1: player1, -1: player2}

    def play_games(self, half_n=5, verbose=0):
        score = 0
        for first_player in [1, -1]:
            for _ in range(half_n):
                score += self.play_game(first_player, verbose=verbose)
        return score, half_n * 2

    def play_game(self, first_player, verbose=0):
        players = {1: self.players[first_player],
                  -1: self.players[-first_player]}

        state = self.game.init_state()

        while True:           
            terminal, value = self.game.terminal_value(state)
            if terminal:
                if verbose > 0:
                    print(self.game.display_state(state))
                return value * state.player * first_player
            
            if verbose > 1:
                print(self.game.display_state(state))
            
            action = players[state.player](state)

            state = self.game.next_state(state, action)