import numpy as np

class CrossEntropy:

    @staticmethod
    def cost(h, label):
        return -(label * np.log(h) + (1-label) * np.log(1-h))

    @staticmethod
    def gradient(h, label):
        return (h - label) / (h * (1-h))
