## 

# Emulating node behaviour for Federated Learning

##
import numpy as np
import logging
from abc import ABC, abstractmethod
from common import cycling_window

log = logging.getLogger(__name__)


class Node:

    _id = 1

    def __init__(self, id, X, y):
        if id == None:
            id = Node._id
            Node._id += 1
        self.id = id
        self.X = X
        self.y = y

    def evaluate_model(self, model, individual_fitness):
        return len(self.y), individual_fitness(model, self.X, self.y)


    def evaluate_multiple_models(self, models, individual_fitness):
        return len(self.y), [individual_fitness(model, self.X, self.y) for model in models]

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





class NodeIteratorBase(ABC):

    def __init__(self, X, y, change_interval):
        self.change_interval = change_interval
        self.nodes = NodeIteratorBase.split_nodes(X, y)
        self._access_nr = 0
        self.current_subset = []

    @staticmethod
    def split_nodes(X, y):
        """
            We will split the data into 1 order of magnitude less nodes than the actual length of data.
            This ensures the 'Massively Distributed' federated property.
        """
        node_count = int(len(y) / 10) 
        log.info('Splitting |%d| data into |%d| nodes', len(y), node_count)
        split_indices = np.append([0, len(X)], np.random.choice(range(1, len(X)), node_count - 1, replace=False))
        split_indices.sort()
        nodes = [Node(None, X[start:end], y[start:end]) for start, end in zip(split_indices[:-1], split_indices[1:])]
        for node in nodes:
            log.info(node)

        return nodes

    @abstractmethod
    def update(self):
        pass

    def __iter__(self):
        while(True):
            yield self.__next__()

    def __next__(self):
        if self._access_nr % self.change_interval == 0:
            self.update()

        self._access_nr += 1

        return self.current_subset


class NodeIteratorRandomSingleNodeSingleElement(NodeIteratorBase):

    def __init__(self, X, y, change_interval):
        super().__init__(X, y, change_interval)


    def update(self):
        """
            Choosing a random node and from that node chose a random sample.
            Create a list of the new single element node. 
        """
        node = np.random.choice(self.nodes)

        idx = np.random.choice(range(len(node.y)))
        X_new_shape = (1,) + node.X.shape[1:]
        y_new_shape = (1,) + node.y.shape[1:]
        X1 = np.array(node.X[idx]).reshape(X_new_shape)
        y1 = np.array(node.y[idx]).reshape(y_new_shape)
        self.current_subset = [Node(None, X1, y1)]


class NodeIteratorRandomSubset(NodeIteratorBase):

    def __init__(self, X, y, change_interval, subset_ratio):
        super().__init__(X, y, change_interval)
        self.subset_ratio = subset_ratio


    def update(self):
        """
            Choosing a random set of nodes.
        """
        self.current_subset = np.random.choice(self.nodes, int(len(self.nodes) * self.subset_ratio))

class NodeIteratorMovingWindow(NodeIteratorBase):


    def __init__(self, X, y, change_interval, window_ratio):
        super().__init__(X, y, change_interval)
        self.cycling_window = cycling_window(self.nodes, int(len(self.nodes) * window_ratio))


    def update(self):
        """
            Cycle through the list of nodes with a moving window.
        """
        self.current_subset = next(self.cycling_window)