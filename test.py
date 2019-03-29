from common import Trial, safe_log, nll

import numpy as np
import pickle

def vectorize_generation(list_of_models):
    generation_vector = []
    for model in list_of_models:
        model_vector = []
        for layer in model:
            layer_vectorized = np.reshape(layer, newshape=(layer.size))
            np.reshape(layer_vectorized, newshape=layer.shape)
            model_vector.extend(layer_vectorized)
        generation_vector.append(model_vector)
    return np.array(generation_vector)

def generation_vector_to_models(generation_vector, list_of_old_models):
    models = []
    model_idx = 0
    for old_model in list_of_old_models:
        start = 0
        end = 0
        model = []
        for old_layer in old_model:
            end = end + old_layer.size
            layer_vector = generation_vector[model_idx][start:end]
            layer = np.reshape(layer_vector, newshape=old_layer.shape)
            model.append(layer) 
            start = end
        models.append(model)
        model_idx = model_idx + 1

    return models

with open('json_weights.pkl', 'rb') as f:
    json, weights = pickle.load(f)

vector_of_gens = vectorize_generation([weights])

generation = generation_vector_to_models(vector_of_gens, [weights])
