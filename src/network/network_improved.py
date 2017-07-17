import numpy as np

from network.network import Network
from cost.cross_entropy import CrossEntropy

class NetworkImproved(Network):

    def __init__(self, size, cost = CrossEntropy):
        self.size = size
        self.cost = cost
        self.layers = len(size)

        # Initialize weights with improved Gaussian function
        self.weights = [np.random.randn(x, y) / np.sqrt(y) for x, y in zip(size[1:], size[:-1])]
        self.biases = [np.random.randn(x) for x in size[1:]]

        # Set default attributes
        self.l2_norm = False
        self.l2_norm_lambda = 1.0

    def setAttribs(self, **kwargs):
        if 'l2_norm' in kwargs.keys():
            self.l2_norm = kwargs['l2_norm']

        if 'l2_norm_lambda' in kwargs.keys():
            self.l2_norm_lambda = kwargs['l2_norm_lambda']

    def back_propergation(self, data, label):
        delta_w, delta_b = super().back_propergation(data, label)

        # L2 regularization
        if self.l2_norm:
            for i in range(len(delta_w)):
                delta_w[i] += self.l2_norm_lambda / self.train_size * self.weights[i]

        return delta_w, delta_b
