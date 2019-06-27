
# based on https://towardsdatascience.com/artificial-neural-networks-optimization-using-genetic-algorithm-with-python-1fe8ed17733e
# different crossover function: mean
from common import load_df, df_to_ML_data, timed_method, ts
from genetic import run_federated_evolution, individual_fitness_f1, initialize_evolution, individual_fitness_nmse
from genetic import federated_population_fitness_model_based
from node import NodeIteratorRandomSingleNodeSingleElement


from IPython.core.display import Javascript
from IPython.display import display
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import pickle
import keras
from keras.models import model_from_json
from keras import metrics
import tensorflow as tf
from tqdm import tqdm_notebook as tqdm
import matplotlib
import matplotlib.pyplot
import logging
import random
import sys


logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',level=logging.INFO, 
                    filename=__file__[:-3] + ts() + '.log', filemode='w+')


log = logging.getLogger(__name__)

if __name__ == "__main__":

    config = tf.ConfigProto()#device_count = {'GPU': 0})
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

    X_train, X_test, y_train, y_test, y_train_argmax, y_test_argmax = df_to_ML_data(load_df())


    node_subset_change_interval = 3
    population_size = 50
    num_parents_mating = 8
    num_generations = 5000
    mutation_chance = 0.01
    mutation_rate = 3
    stuck_multiplier = 1
    stuck_evasion_rate = 1.25
    stuck_multiplier_max = 5
    stuck_check_length = 30
    save_interval = 5
    plot_interval = 150000
    federated_population_fitness = federated_population_fitness_model_based
    individual_fitness = individual_fitness_nmse



    generation_start, best_fitness_of_each_generation, best_accuracy_of_each_generation, best_model_of_each_generation,\
        population_weights = initialize_evolution(__file__, population_size)

    nodes_iterator = NodeIteratorRandomSingleNodeSingleElement(X_train, y_train, node_subset_change_interval)


    run_federated_evolution(nodes_iterator=nodes_iterator,\
                        X_validate=X_test, y_validate=y_test,\
                        num_parents_mating=num_parents_mating, num_generations=num_generations,\
                        federated_population_fitness=federated_population_fitness, individual_fitness=individual_fitness,\
                        generation_start=generation_start, mutation_chance=mutation_chance, mutation_rate=mutation_rate,\
                        best_fitness_of_each_generation=best_fitness_of_each_generation, best_accuracy_of_each_generation=best_accuracy_of_each_generation,\
                        best_model_of_each_generation=best_model_of_each_generation, population_weights=population_weights,\
                        plot_interval=plot_interval, stuck_multiplier=stuck_multiplier, stuck_multiplier_max=stuck_multiplier_max,\
                        save_interval=save_interval, stuck_evasion_rate=stuck_evasion_rate, stuck_check_length=stuck_check_length,\
                        checkpoint_filename=__file__)



