import numpy as np
import weakref
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def to_onehot(labels, num_classes=10):
    return np.eye(num_classes)[labels]

# ----------------------------
# 1. Variable & Function Core
# ----------------------------
class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        seen = set()
        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad += gx

                if x.creator is not None and x.creator not in seen:
                    funcs.append(x.creator)
                    seen.add(x.creator)


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(y) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]
        return outputs[0] if len(outputs) == 1 else outputs

    def forward(self, *xs):
        raise NotImplementedError()

    def backward(self, *gys):
        raise NotImplementedError()


# ----------------------------
# 2. Basic Functions
# ----------------------------
class MatMul(Function):
    def forward(self, x, W):
        return x @ W

    def backward(self, gy):
        x, W = self.inputs
        gy = np.asarray(gy)  # gy를 numpy 배열로 변환
        W_data = np.asarray(W.data)  # W.data를 numpy 배열로 변환
        x_data = np.asarray(x.data)  # x.data를 numpy 배열로 변환

        # 차원 확인 및 디버깅용 출력
        assert gy.shape[1] == W_data.T.shape[0], f"gy와 W.T의 차원이 맞지 않습니다: {gy.shape}, {W_data.T.shape}"
        assert x_data.T.shape[1] == gy.shape[0], f"x.T와 gy의 차원이 맞지 않습니다: {x_data.T.shape}, {gy.shape}"

        return gy @ W_data.T, x_data.T @ gy

class Add(Function):
    def forward(self, x, b):
        return x + b

    def backward(self, gy):
        return gy, np.sum(gy, axis=0)


class ReLU(Function):
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, gy):
        return gy * self.mask


class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):  # t는 now one-hot array
        self.t = t
        x = np.asarray(x)  # x를 numpy 배열로 변환
        self.y = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.y /= np.sum(self.y, axis=1, keepdims=True)
        log_likelihood = -np.sum(t * np.log(self.y + 1e-7), axis=1)
        return np.mean(log_likelihood)

    def backward(self, gy=1.0):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t.data) / batch_size
        return dx, None

# ----------------------------
# 3. Layer & MLP Model
# ----------------------------
class Dense:
    def __init__(self, in_dim, out_dim):
        self.W = Variable(np.random.randn(in_dim, out_dim) * 0.01)
        self.b = Variable(np.zeros(out_dim))

    def __call__(self, x):
        return Add()(MatMul()(x, self.W), self.b)

    def params(self):
        return [self.W, self.b]


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


# ----------------------------
# 4. Evaluator
# ----------------------------
class Evaluator:
    def __init__(self, model):
        self.model = model

    def accuracy(self, Xs, ys):
        inputs = Variable(Xs)
        outputs = self.model(inputs)
        y_pred = np.asarray(outputs.data).argmax(axis=1)  # numpy 배열로 변환
        return np.mean(y_pred == ys)

# ----------------------------
# 5. Load MNIST
# ----------------------------
def load_mnist(normalize=True):
    mnist = fetch_openml('mnist_784', version=1)
    x = mnist['data'].astype(np.float32)
    y = mnist['target'].astype(np.int32)
    if normalize:
        x /= 255.0
    return x[:60000], y[:60000], x[60000:], y[60000:]


# ----------------------------
# 6. Training & Evaluation
# ----------------------------
if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_mnist()
    model = MLP()
    evaluator = Evaluator(model)
    lr = 0.1
    batch_size = 100

    # 한 epoch만 학습
    for i in range(0, len(x_train), batch_size):
        labels = y_train[i:i + batch_size]
        xb = Variable(x_train[i:i + batch_size])
        yb = Variable(to_onehot(labels))  # ← now one-hot
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