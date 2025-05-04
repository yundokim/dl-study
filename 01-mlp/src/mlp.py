from core import Variable, SoftmaxCrossEntropy, ReLU, to_onehot, Model
from layers import Dense
from training import Evaluator, SGD, train
from data import load_mnist

class MLP(Model):
    def __init__(self):
        super().__init__()
        self.layers = [
            Dense(784, 128),
            ReLU(),
            Dense(128, 10)
        ]

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_mnist()
    model = MLP()
    optimizer = SGD(model.params(), lr=0.1)
    evaluator = Evaluator(model)

    train(model, optimizer, x_train, y_train, batch_size=100)

    train_acc = evaluator.accuracy(x_train, y_train)
    test_acc = evaluator.accuracy(x_test, y_test)

    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test  accuracy: {test_acc:.4f}")
