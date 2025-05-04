from core import Variable, MatMul, Add
import numpy as np

class Dense:
    def __init__(self, in_dim, out_dim):
        self.W = Variable(np.random.randn(in_dim, out_dim) * 0.01)
        self.b = Variable(np.zeros(out_dim))

    def __call__(self, x):
        return Add()(MatMul()(x, self.W), self.b)

    def params(self):
        return [self.W, self.b]