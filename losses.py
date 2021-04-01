import numpy as np

class Loss:
    def loss(self, x: float, y: float) -> float: raise NotImplementedError
    def grad(self, x: float, y: float) -> float: raise NotImplementedError

class SQUARE(Loss):
    def loss(self, x: float, y: float) -> float: return .5 * np.square(x - y)
    def grad(self, x: float, y: float) -> float: return x - y
