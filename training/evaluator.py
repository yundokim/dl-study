import numpy as np
from core import Variable

class Evaluator:
    def __init__(self, model):
        self.model = model

    def accuracy(self, Xs, ys):
        inputs = Variable(Xs)
        outputs = self.model(inputs)
        y_pred = np.asarray(outputs.data).argmax(axis=1)
        return np.mean(y_pred == ys)