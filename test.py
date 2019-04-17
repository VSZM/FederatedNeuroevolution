"""
2-input XOR example -- this is most likely the simplest possible example.
"""

from common import load_df, df_to_ML_data
from myneat import LoggingReporter
from sklearn.metrics import accuracy_score
import os
import pickle
import neat
import neat_visualization as visualize
import logging
import sys
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',level=logging.INFO, 
                    filename='genetic_neat.log', filemode='w+')


log = logging.getLogger(__name__)

X_train, X_test, y_train, y_test = df_to_ML_data(load_df())


def eval_genomes(genomes, config):
    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        y_pred = [net.activate(Xi) for Xi in X_train]
        genome.fitness = accuracy_score(y_train, y_pred)


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(LoggingReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    log.info('\nBest genome:\n%s', winner)

    node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)


if __name__ == '__main__':
    run('genetic_neat.ini')