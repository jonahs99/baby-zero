import numpy as np
import neat

class NN:
    def __init__(self, game, config_file):
        self.game = game
        
        # Load configuration.
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_file)

        # Create the population, which is the top-level object for a NEAT run.
        self.pop = neat.Population(self.config)
        self.winner_genome = None

    def predictor(self):
        if self.winner_genome is None:
            return None

        def fn(state):
            winner_net = neat.nn.FeedForwardNetwork.create(self.winner_genome, self.config)
            output = winner_net.activate(self.game.inputs(state))

            # Probablities, Value estimate
            return output[:-1], output[-1]
        return fn

    def train(self, examples, generations=10):
        # Examples of the form [inputs, pi, v]
        def eval_genomes(genomes, config):
            for genome_id, genome in genomes:
                genome.fitness = 0
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                for inputs, pi, v in examples:
                    output = net.activate(inputs)
                    # Weight the v, pi error equaly
                    genome.fitness -= np.mean((output[:-1] - pi) ** 2)
                    genome.fitness -= (output[-1] - v) ** 2
        self.winner_genome = self.pop.run(eval_genomes, generations)