## 

# Emulating node behaviour for Federated Learning

##
import numpy as np



class Node:

    _id = 1

    def __init__(self, id, X, y, reachability_treshold):
        if id == None:
            id = Node._id
            Node._id += 1
        self.id = id
        self.X = X
        self.y = y
        self.reachability_treshold = reachability_treshold

    def is_reachable(self, stimuli):
        return stimuli > self.reachability_treshold

    def evaluate_model(self, model, individual_fitness):
        return len(self.y), individual_fitness(model, self.X, self.y)


    def evaluate_multiple_models(self, models, individual_fitness):
        return len(self.y), [individual_fitness(model, self.X, self.y) for model in models]

    def evaluate_multiple_models_random_item(self, models, individual_fitness):
        idx = np.random.choice(range(len(self.y)))
        X_new_shape = (1,) + self.X.shape[1:]
        y_new_shape = (1,) + self.y.shape[1:]
        X1 = np.array(self.X[idx]).reshape(X_new_shape)
        y1 = np.array(self.y[idx]).reshape(y_new_shape)
        return 1, [individual_fitness(model, X1, y1) for model in models]

    def __eq__(self, other):
            if other == None:
                return False

            myId = self.id
            otherId = other.id
            if myId == otherId:
                return True
            else:
                return False

    def __str__(self):
        # Non alcoholic: [1, 0], Alcoholic: [0, 1]
        argmaxed = np.argmax(self.y, axis = 1)
        alcoholic_count = np.count_nonzero(argmaxed == 1)
        non_alcoholic_count = np.count_nonzero(argmaxed == 0)

        return "Node(id = |%d|, Sample count = |%d|, Alcoholic samples = |%d|, Non-Alcoholic samples = |%d|)" %\
            (self.id, len(self.y), alcoholic_count, non_alcoholic_count)

    __repr__ = __str__





