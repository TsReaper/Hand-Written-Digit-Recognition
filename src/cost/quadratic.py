class Quadratic:

    @staticmethod
    def cost(h, label):
        return (h - label) ** 2.0

    @staticmethod
    def gradient(h, label):
        return 2.0 * (h - label)
