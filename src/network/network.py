import numpy as np
import json

import utils.math as math
from cost.quadratic import Quadratic

class Network:

    def __init__(self, size, cost = Quadratic):
        self.size = size
        self.cost = cost
        self.layers = len(size)

        self.weights = [np.random.randn(x, y) for x, y in zip(size[1:], size[:-1])]
        self.biases = [np.random.randn(x) for x in size[1:]]

    def predict(self, data):
        for i in range(self.layers-1):
            data = math.sigmoid(np.dot(self.weights[i], data) + self.biases[i])
        return data

    def train(self, data_set, label_set, learn_rate, epoch, batch_size):
        self.train_size = len(data_set)

        for i in range(epoch):
            # Train network in batches
            for j in range(0, self.train_size, batch_size):
                if j + batch_size >= self.train_size:
                    batch_num = self.train_size - j
                else:
                    batch_num = batch_size

                # Stochastic gradient descent
                delta_w, delta_b = self.process_batch(data_set[j:j + batch_num], label_set[j:j + batch_num])
                self.weights = [x - learn_rate/batch_num * y for x, y in zip(self.weights, delta_w)]
                self.biases = [x - learn_rate/batch_num * y for x, y in zip(self.biases, delta_b)]

            print('Epoch %d/%d' % (i+1, epoch))

    def process_batch(self, data_set, label_set):
        delta_w = [np.zeros((x, y)) for x, y in zip(self.size[1:], self.size[:-1])]
        delta_b = [np.zeros((x)) for x in self.size[1:]]

        for i in range(len(data_set)):
            tmp_w, tmp_b = self.back_propergation(data_set[i], label_set[i])
            delta_w = [x + y for x, y in zip(delta_w, tmp_w)]
            delta_b = [x + y for x, y in zip(delta_b, tmp_b)]

        return delta_w, delta_b

    def back_propergation(self, data, label):
        # Calculate input and output for all neurons
        a, z = [data], []
        for i in range(self.layers-1):
            z.append(np.dot(self.weights[i], a[-1]) + self.biases[i])
            a.append(math.sigmoid(z[-1]))

        # Back propergation
        delta_w = [None for i in range(self.layers-1)]
        delta_b = [None for i in range(self.layers-1)]
        error = self.cost.gradient(a[-1], label) * math.sigmoid_diff(z[-1])

        for i in range(self.layers-2, -1, -1):
            delta_b[i] = error
            delta_w[i] = np.array(np.mat(error).T * np.mat(a[i]))
            if i > 0:
                error = np.dot(self.weights[i].T, error) * math.sigmoid_diff(z[i-1])

        return delta_w, delta_b

    def import_json(self, s):
        d = json.loads(s)
        self.size = d['size']
        self.weights = list(map(lambda x: np.array(x), d['weights']))
        self.biases =list(map(lambda x: np.array(x), d['biases']))

    def get_json(self):
        size = self.size
        w = list(map(lambda x: x.tolist(), self.weights))
        b = list(map(lambda x: x.tolist(), self.biases))
        d = {'size': size, 'weights': w, 'biases': b}
        return json.dumps(d)
