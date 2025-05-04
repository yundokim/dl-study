from core import Variable, Add, MatMul, ReLU, SoftmaxCrossEntropy, to_onehot
from layers import Dense
from training import Evaluator
from data import load_mnist


class MLP:
    def __init__(self):
        self.layers = [
            Dense(784, 128),
            ReLU(),
            Dense(128, 10)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x) if isinstance(layer, Dense) else layer(x)
        return x

    def params(self):
        p = []
        for layer in self.layers:
            if isinstance(layer, Dense):
                p.extend(layer.params())
        return p


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_mnist()
    model = MLP()
    evaluator = Evaluator(model)
    lr = 0.1
    batch_size = 100

    for i in range(0, len(x_train), batch_size):
        xb = Variable(x_train[i:i + batch_size])
        labels = y_train[i:i + batch_size]
        yb = Variable(to_onehot(labels))
        out = model(xb)
        loss = SoftmaxCrossEntropy()(out, yb)
        loss.backward()

        for p in model.params():
            p.data -= lr * p.grad
            p.grad = None

    train_acc = evaluator.accuracy(x_train, y_train)
    test_acc = evaluator.accuracy(x_test, y_test)

    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test  accuracy: {test_acc:.4f}")