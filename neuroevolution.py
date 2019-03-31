import numpy as np


def vectorize_population(population_of_models):
    population_vector = []
    for model in population_of_models:
        model_vector = []
        for layer in model:
            layer_vectorized = np.reshape(layer, newshape=(layer.size))
            np.reshape(layer_vectorized, newshape=layer.shape)
            model_vector.extend(layer_vectorized)
        population_vector.append(model_vector)
    return np.array(population_vector)

def population_vector_to_models(population_vector, population_of_old_models):
    models = []
    model_idx = 0
    for old_model in population_of_old_models:
        start = 0
        end = 0
        model = []
        for old_layer in old_model:
            end = end + old_layer.size
            layer_vector = population_vector[model_idx][start:end]
            layer = np.reshape(layer_vector, newshape=old_layer.shape)
            model.append(layer) 
            start = end
        models.append(model)
        model_idx = model_idx + 1

    return models