import numpy as np
import weakref
from core import Variable

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