import numpy as np

class Model:
    def __init__(self):
        self.layers = []

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def params(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'params'):
                params.extend(layer.params())
        return params

    def save(self, path):
        params = self.params()
        weights = {f"param_{i}": p.data for i, p in enumerate(params)}
        np.savez(path, **weights)

    def load(self, path):
        loaded = np.load(path)
        for i, p in enumerate(self.params()):
            p.data = loaded[f"param_{i}"]
