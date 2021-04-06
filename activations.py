import numpy as np

class Activation:
    def func(self, x: float) -> float: raise NotImplementedError
    def grad(self, x: float) -> float: raise NotImplementedError

class RELU(Activation):
    def func(self, x: float) -> float: return np.max([0, x])
    def grad(self, x: float) -> float: return 0 if x <= 0 else 1

class LINEAR(Activation):
    def func(self, x: float) -> float: return x
    def grad(self, x: float) -> float: return 1.

class TANH(Activation):
    def func(self, x: float) -> float: return np.tanh(x)
    def grad(self, x: float) -> float: return 1. - np.tanh(x) * np.tanh(x)

class SIGMOID(Activation):
    def func(self, x: float) -> float: return 1. / (1 + np.exp(-x))
    def grad(self, x: float) -> float: return 1. / (1 + np.exp(-x)) * \
             (1 - 1 / (1 + np.exp(-x)))
