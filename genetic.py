

# based on https://towardsdatascience.com/artificial-neural-networks-optimization-using-genetic-algorithm-with-python-1fe8ed17733e
# different crossover function: mean
from common import Trial, safe_log, nll, load_df, df_to_ML_data, timed_method


from IPython.core.display import Javascript
from IPython.display import display
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import pickle
import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from keras import backend as K
from keras.models import model_from_json
from keras.layers import Dense, Flatten, BatchNormalization, Dropout, Lambda
from keras.layers import Conv2D, AveragePooling2D
from keras.models import Sequential
import keras_metrics
from keras import metrics
import tensorflow as tf
from tqdm import tqdm_notebook as tqdm
import matplotlib
import matplotlib.pyplot
import logging
import random
import sys
from line_profiler import LineProfiler

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',level=logging.INFO, 
                    filename='genetic_learning_weights_only_kernel.log', filemode='a')


log = logging.getLogger(__name__)


config = tf.ConfigProto()#device_count = {'GPU': 0})
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))



with open('json_weights_bn_simple.pkl', 'rb') as f:
    model_topology_json, weights = pickle.load(f)

(model_topology_json, weights)



X_train, X_test, y_train, y_test = df_to_ML_data(load_df())



def check_weights(a, b):
    assert len(a) == len(b)
    for i in range(len(a)):
        assert a[i].shape ==  b[i].shape

def create_model_programatically(weights = None):
    input_shape = (64, 256, 1)
    num_classes = 2
    
    model = Sequential()
    model.add(Conv2D(30, kernel_size=(1, 25),
                     input_shape=input_shape))
    model.add(BatchNormalization(momentum=0.1))
    model.add(Conv2D(10, kernel_size=(64, 1)))
    model.add(BatchNormalization(momentum=0.1))
    model.add(Lambda(lambda x: x ** 2))
    model.add(AveragePooling2D(pool_size=(1, 15), strides=(1, 1)))
    model.add(Lambda(lambda x: safe_log(x)))
    model.add(BatchNormalization(momentum=0.1))
    #model.add(Dropout(0.5))
    model.add(Conv2D(2, kernel_size=(1, 8), dilation_rate=(15, 1)))
    model.add(BatchNormalization(momentum=0.1))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))


    if weights != None:
        #check_weights(model.get_weights(), weights)

        model.set_weights(weights)

    #model.compile(#loss=keras.losses.categorical_crossentropy,
              #optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
     #         metrics=[keras_metrics.binary_f1_score(), metrics.binary_accuracy])
                        #metrics.binary_accuracy, metrics.cosine_proximity, metrics.mean_absolute_error, 
                       #metrics.mean_absolute_percentage_error, metrics.mean_squared_error,
                       #keras_metrics.precision(), keras_metrics.recall(), keras_metrics.binary_f1_score(), 
                       #keras_metrics.binary_false_negative(), keras_metrics.binary_false_positive(),
                       #keras_metrics.binary_true_negative(), keras_metrics.binary_true_positive()
                        #])

    
    return model

def individual_fitness(model_topology_json, model_weights, X, y):
    #keras_model = model_from_json(model_topology_json)
    keras_model = create_model_programatically(model_weights)
    try:
        return mean_absolute_error(y, keras_model.predict(X, batch_size=512))
    except:
        return 100

def population_fitness(model_topology_json, population_of_models, X, y):
    fitness_scores = []
    
    for model_weights in population_of_models:#tqdm(population_of_models, desc='Current Population Fitness calculation progress', position=2):
        fitness_scores.append(individual_fitness(model_topology_json, model_weights, X, y))
                          
    return fitness_scores

def fittest_parents_of_generation(population, fitness_scores, num_parents, selector = np.argmin):
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
def crossover(parent_models, offsprings_size):
    offsprings = []
    for k in range(offsprings_size):
        # Index of the first parent to mate.
        parent1_idx = k % len(parent_models)
        # Index of the second parent to mate.
        parent2_idx = (k+1) % len(parent_models)
        # mix of each modell kernelwise
        offsprings.append(kernelwise_mix(parent_models[parent1_idx], parent_models[parent2_idx]))
    return offsprings

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
            mutation_value = np.random.uniform(-1.0 * mutation_rate, 1.0 * mutation_rate, 1)
            layer[mutation_indices] += mutation_value
        
    return offsprings




def do_stuff():

    population_size = 20 #sol_per_pop
    num_parents_mating = 8
    num_generations = 5
    mutation_chance = 0.1
    mutation_rate = 2
    reference_weights = weights

    #Creating the initial population.
    population_weights = []
    for _ in range(0, population_size):
        population_weights.append(create_model_programatically().get_weights())
        
    K.clear_session()



    individual_fitness(None, population_weights[2], X_test, y_test)
    individual_fitness(None, population_weights[3], X_test, y_test)



    best_of_each_generation = []

    for generation in tqdm(range(num_generations), desc='Generations progress', position=1):

        log.debug('Testing generation |%d| population: |%s|', generation, population_weights)
        # Measuring the fitness of each chromosome in the population.
        fitness_scores = population_fitness(None, population_weights, X_test, y_test)
        
        best_of_this_generation = min(fitness_scores)
        best_of_each_generation.append(best_of_this_generation)
        log.info("Best of geration |%d| has accuracy of |%f|", generation, best_of_this_generation)
        log.info('Fitness scores of this generation: |%s|', fitness_scores)
        
        
        # Selecting the best parents in the population for mating.
        parents = fittest_parents_of_generation(population_weights.copy(), fitness_scores, num_parents_mating)

        # Generating next generation using crossover.
        offsprings = crossover(parents, len(population_weights) - num_parents_mating)

        # Adding some variations to the offsrping using mutation.
        offsprings = mutation(offsprings, mutation_chance=mutation_chance, mutation_rate=mutation_rate)

        # Creating the new generation based on the parents and offspring.
        population_weights = []
        population_weights.extend(parents)
        population_weights.extend(offsprings)

        #cleanup resources
        K.clear_session()
        




lp = LineProfiler()
lp.add_function(population_fitness)
lp.add_function(individual_fitness)
lp.add_function(create_model_programatically)
lp_wrapper = lp(do_stuff)
lp_wrapper()
lp.print_stats()