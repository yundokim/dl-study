from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np

def load_mnist(normalize=True):
    mnist = fetch_openml('mnist_784', version=1)
    x = mnist['data'].astype(np.float32)
    y = mnist['target'].astype(np.int32)
    if normalize:
        x /= 255.0
    return x[:60000], y[:60000], x[60000:], y[60000:]