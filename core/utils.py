import numpy as np

def to_onehot(labels, num_classes=10):
    return np.eye(num_classes)[labels]