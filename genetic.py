
from keras.layers import Dense, Flatten, BatchNormalization, Dropout, Lambda
from keras.layers import Conv2D, AveragePooling2D
from keras.models import Sequential
from keras import backend as K
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import pickle
from common import safe_log, plot_learning, cycling_window
from node import Node
import numpy as np
import os
from more_itertools import peekable
import collections
import logging

log = logging.getLogger(__name__)


try:
    get_ipython
    from tqdm import tqdm_notebook as tqdm
    log.debug('Using tqdm notebook version')
except:
    from tqdm import tqdm
    log.debug('Using tqdm console version')



def check_weights(a, b):
    assert len(a) == len(b)
    for i in range(len(a)):
        assert a[i].shape ==  b[i].shape

def create_model_programatically(weights = None):
    K.clear_session()

    input_shape = (64, 256, 1)
    num_classes = 2
    
    model = Sequential()
    model.add(Conv2D(30, kernel_size=(1, 25),
                     input_shape=input_shape))
    model.add(Conv2D(10, kernel_size=(64, 1)))
    model.add(Lambda(lambda x: x ** 2))
    model.add(AveragePooling2D(pool_size=(1, 15), strides=(1, 1)))
    model.add(Lambda(lambda x: safe_log(x)))
    model.add(Conv2D(2, kernel_size=(1, 8), dilation_rate=(15, 1)))
    model.add(BatchNormalization(momentum=0.1))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))


    if weights != None:
        #check_weights(model.get_weights(), weights)

        model.set_weights(weights)

    
    return model


#@profile
def individual_accuracy(keras_model, X, y):
    y_pred = keras_model.predict_classes(X, batch_size=512)
    
    return accuracy_score(y, y_pred)


def individual_fitness_f1(keras_model, X, y):
    y_pred = keras_model.predict_classes(X, batch_size=512)


    return f1_score(y, y_pred)

#@profile
def individual_fitness_nmse(keras_model, X, y):
    y_pred = keras_model.predict(X, batch_size=512)

    try:
        return -1.0 * mean_squared_error(y, y_pred)
    except:
        log.error("Error with mse calculation! Inputs: |%s| and |%s|", y, y_pred, exc_info=True)
        return -100000


# Randomly select a node which runs the evaluation on a random data item
#@profile
def federated_population_fitness_single_node_singe_item(nodes, individual_fitness, population_of_models):
    node_stimuli = np.random.uniform(0, 1, len(nodes))
    
    keras_models = [create_model_programatically(model) for model in population_of_models]
    reachable_nodes = [nodes[np.argmax(node_stimuli)]]
    weights_and_scores = np.array([node.evaluate_multiple_models_random_item(keras_models, individual_fitness)\
                            for node in tqdm(reachable_nodes, desc='Fitness progress', position=2)]).transpose()

    return weights_and_scores[1][0]

#@profile
def fitness_of_model_for_nodes(nodes, model_weights, individual_fitness):
    keras_model = create_model_programatically(model_weights)
    weights_and_scores = np.array([node.evaluate_model(keras_model, individual_fitness) for node in nodes]).transpose()

    return np.average(weights_and_scores[1], weights=weights_and_scores[0], axis = 0)

def federated_population_fitness_model_based_all_nodes(nodes, individual_fitness, population_of_models):
    return federated_population_fitness_model_based(nodes, individual_fitness, population_of_models, 1, 1)

# Based on models. For each model get the fitness from randomly selected nodes. 
#@profile
def federated_population_fitness_model_based(nodes, individual_fitness, population_of_models, min_stimuli = 0, max_stimuli = 1):
    if isinstance(nodes, collections.Iterator):
        nodes = next(nodes)
    
    node_stimuli = np.random.uniform(min_stimuli, max_stimuli, len(nodes))
    
    #weights_and_scores = [node.evaluate(keras_models, individual_fitness) for node, stimuli in zip(nodes, node_stimuli) if node.is_reachable(stimuli)]
    reachable_nodes = [node for node, stimuli in zip(nodes, node_stimuli) if node.is_reachable(stimuli)]
    fitness_scores = [fitness_of_model_for_nodes(reachable_nodes, model, individual_fitness) for model in\
                        tqdm(population_of_models, desc='Fitness progress', position=2)]

    return fitness_scores


def population_fitness(individual_fitness, population_of_models, X, y):
    return [individual_fitness(create_model_programatically(model_weights), X, y) for model_weights in tqdm(population_of_models, desc='Fitness progress', position=2)]

#@profile
def fittest_parents_of_generation(population, fitness_scores, num_parents, selector = np.argmax):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = []
    for _ in range(num_parents - 1):
        best_fitness_idx = selector(fitness_scores)
        parents.append(population[best_fitness_idx])
        del population[best_fitness_idx]
        del fitness_scores[best_fitness_idx]
    
    # mixing in a lucky one.. because sometimes anyone can get lucky ;)
    np.random.shuffle(population)
    parents.append(population[0])

    return parents


# Mixing too models by keeping their kernel weights intact
#@profile
def kernelwise_mix(model_a, model_b):
    mix = []
    for i in range(len(model_a)):
        layer_a = model_a[i]
        layer_b = model_b[i]        
        # choosing kernels
        choice = np.random.randint(2, size = int(layer_a.size / layer_a.shape[-1])).reshape(layer_a.shape[:-1]).astype(bool)
        # extending the chosen kernel bools to the level of single values
        choice = np.repeat(choice, layer_a.shape[-1]).reshape(layer_a.shape)

        layer_mix = np.where(choice, layer_a, layer_b)
        mix.append(layer_mix)
        
    return mix

# Creates offsprings by mixing the layers of the model weights
#@profile
def crossover(parent_models, offsprings_size):
    offsprings = []
    np.random.shuffle(parent_models)
    for k in range(offsprings_size):
        # Index of the first parent to mate.
        parent1_idx = k % len(parent_models)
        # Index of the second parent to mate.
        parent2_idx = (k+1) % len(parent_models)
        # mix of each modell kernelwise
        offsprings.append(kernelwise_mix(parent_models[parent1_idx], parent_models[parent2_idx]))
    return offsprings

#@profile
def mutation(offsprings, mutation_chance=0.1, mutation_rate=1):

    # Mutation changes a single gene in each offspring randomly.
    for offspring in offsprings:
        for layer in offspring:
            trues = np.full(int(layer.size * mutation_chance), True)
            falses = np.full(layer.size - trues.size, False)
            mutation_indices = np.append(trues, falses)
            np.random.shuffle(mutation_indices)
            mutation_indices = mutation_indices.reshape(layer.shape)
                
            # The random value to be added to the gene.
            mutation_multiplier = np.random.normal(loc=0.0, scale=0.01 * mutation_rate, size=1)
            layer[mutation_indices] = layer[mutation_indices] + layer[mutation_indices] * mutation_multiplier
        
    return offsprings

#@profile
def save_state(checkpoint_filename, best_fitness_of_each_generation, best_accuracy_of_each_generation, best_model_of_each_generation, population_weights):
    # Saving weights
    with open(checkpoint_filename + '.checkpoint','wb') as f:
        pickle.dump((best_fitness_of_each_generation, best_accuracy_of_each_generation, best_model_of_each_generation, population_weights), f)


def initialize_evolution(checkpoint_filename, population_size):
    if os.path.isfile(checkpoint_filename + '.checkpoint'):
            log.info('Resuming from previous checkpoint')
            with open(checkpoint_filename + '.checkpoint', 'rb') as f:
                best_fitness_of_each_generation, best_accuracy_of_each_generation, best_model_of_each_generation, population_weights = pickle.load(f)
    else:
        log.info('Creating random population')
        population_weights = []
        best_fitness_of_each_generation = []
        best_accuracy_of_each_generation = []
        best_model_of_each_generation = []
        for _ in range(0, population_size):
            population_weights.append(create_model_programatically().get_weights())

        K.clear_session()

    generation_start = len(best_fitness_of_each_generation)

    return generation_start, best_fitness_of_each_generation, best_accuracy_of_each_generation, best_model_of_each_generation, population_weights


#@profile
def run_federated_evolution(*, node_count, node_activation_chance, node_alternative_iterator, X_train, y_train, X_validate, y_validate,\
                    num_parents_mating, num_generations, federated_population_fitness, individual_fitness,\
                    generation_start,mutation_chance,mutation_rate,\
                    best_fitness_of_each_generation, best_accuracy_of_each_generation, best_model_of_each_generation, population_weights,\
                    plot_interval, stuck_multiplier, stuck_multiplier_max, save_interval, stuck_evasion_rate, stuck_check_length, checkpoint_filename):

    log.info('Splitting the training data (size |%d|) into |%d| node data', len(y_train), node_count)
    split_indices = np.append([0, len(X_train)], np.random.choice(range(1, len(X_train)), node_count - 1, replace=False))
    split_indices.sort()
    nodes = [Node(None, X_train[start:end], y_train[start:end], 1 - node_activation_chance) for start, end in zip(split_indices[:-1], split_indices[1:])]
    y_validate_argmax = np.argmax(y_validate, axis = 1)
    for node in nodes:
        log.info(node)

    if node_alternative_iterator is not None:
        nodes = node_alternative_iterator(nodes)

    for generation in tqdm(range(generation_start, num_generations), desc='Evolution progress', position=1):

        log.info('Testing generation |%d|', generation)
        log.debug('Population: |%s|', population_weights)
        # Measuring the fitness of each individual in the population.
        fitness_scores = federated_population_fitness(nodes, individual_fitness, population_weights)
        log.info('Fitness scores of this generation: |%s|', fitness_scores)

        best_fitness_of_each_generation.append(max(fitness_scores))
        best_model_keras = create_model_programatically(population_weights[np.argmax(fitness_scores)])
        best_accuracy_of_each_generation.append(individual_accuracy(best_model_keras, X_validate, y_validate_argmax))
        best_model_of_each_generation.append(population_weights[np.argmax(fitness_scores)])
        log.info("Best of geration |%d| has accuracy of |%f| and fitness_score of |%f|",\
                generation, best_accuracy_of_each_generation[generation], best_fitness_of_each_generation[generation])


        # Selecting the best parents in the population for mating.
        parents = fittest_parents_of_generation(population_weights.copy(), fitness_scores, num_parents_mating)

        # Generating next generation using crossover.
        offsprings = crossover(parents.copy(), len(population_weights) - num_parents_mating)

        stuck_multiplier_value = max(stuck_multiplier, stuck_multiplier_max)
        # Adding some variations to the offsrping using mutation.
        offsprings = mutation(offsprings, mutation_chance=mutation_chance * np.sqrt(stuck_multiplier_value), mutation_rate=mutation_rate * stuck_multiplier_value)

        # Creating the new generation based on the parents and offspring.
        population_weights = []
        population_weights.extend(parents)
        population_weights.extend(offsprings)

        
        # If our accuracy is not increasing we try and speed up mutation 
        if generation > 0 and best_accuracy_of_each_generation[generation] in best_accuracy_of_each_generation[generation-stuck_check_length:generation]:
            stuck_multiplier *= stuck_evasion_rate
            log.info('Stuck at local maximum, expanding mutation rate and chance by stuck multiplier of |%f|', stuck_multiplier)
        else:
            stuck_multiplier = 1

        if generation % plot_interval == 0:
            plot_learning(best_accuracy_of_each_generation, num_generations)


        if generation % save_interval == 0 or generation + 1 == num_generations:
            save_state(checkpoint_filename, best_fitness_of_each_generation, best_accuracy_of_each_generation, best_model_of_each_generation, population_weights)

        #cleanup resources
        K.clear_session()

    save_state(checkpoint_filename, best_fitness_of_each_generation, best_accuracy_of_each_generation, best_model_of_each_generation, population_weights)
    


def run_evolution(*, X_train, y_train,y_acc,\
                    num_parents_mating, num_generations, population_fitness, individual_fitness, generation_start,mutation_chance,mutation_rate,\
                    best_fitness_of_each_generation, best_accuracy_of_each_generation, best_model_of_each_generation, population_weights,\
                    plot_interval, stuck_multiplier, stuck_multiplier_max, save_interval, stuck_evasion_rate, stuck_check_length, checkpoint_filename):
        
    for generation in tqdm(range(generation_start, num_generations), desc='Generations progress', position=1):

        log.info('Testing generation |%d|', generation)
        log.debug('Population: |%s|', population_weights)
        # Measuring the fitness of each individual in the population.
        fitness_scores = population_fitness(individual_fitness, population_weights, X_train, y_train)
        log.info('Fitness scores of this generation: |%s|', fitness_scores)

        best_fitness_of_each_generation.append(max(fitness_scores))
        best_accuracy_of_each_generation.append(individual_accuracy(population_weights[np.argmax(fitness_scores)], X_train, y_acc))
        best_model_of_each_generation.append(population_weights[np.argmax(fitness_scores)])
        log.info("Best of geration |%d| has accuracy of |%f| and fitness_score of |%f|",\
                generation, best_accuracy_of_each_generation[generation], best_fitness_of_each_generation[generation])


        # Selecting the best parents in the population for mating.
        parents = fittest_parents_of_generation(population_weights.copy(), fitness_scores, num_parents_mating)

        # Generating next generation using crossover.
        offsprings = crossover(parents.copy(), len(population_weights) - num_parents_mating)

        stuck_multiplier_value = max(stuck_multiplier, stuck_multiplier_max)
        # Adding some variations to the offsrping using mutation.
        offsprings = mutation(offsprings, mutation_chance=mutation_chance * np.sqrt(stuck_multiplier_value), mutation_rate=mutation_rate * stuck_multiplier_value)

        # Creating the new generation based on the parents and offspring.
        population_weights = []
        population_weights.extend(parents)
        population_weights.extend(offsprings)

        # If our accuracy is not increasing we try and speed up mutation 
        if generation > 0 and best_accuracy_of_each_generation[generation] in best_accuracy_of_each_generation[generation-stuck_check_length:]:
            stuck_multiplier *= stuck_evasion_rate
            log.info('Stuck at local maximum, expanding mutation rate and chance by stuck multiplier of |%f|', stuck_multiplier)
        else:
            stuck_multiplier = 1

        if generation % plot_interval == 0:
            plot_learning(best_accuracy_of_each_generation, num_generations)


        if generation % save_interval == 0 or generation + 1 == num_generations:
            save_state(checkpoint_filename, best_fitness_of_each_generation, best_accuracy_of_each_generation, best_model_of_each_generation, population_weights)

        #cleanup resources
        K.clear_session()
