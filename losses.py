import numpy as np

class Loss:
    def loss(self, ŷ: float, y: float) -> float: raise NotImplementedError
    def grad(self, ŷ: float, y: float) -> float: raise NotImplementedError

class SQUARE(Loss):
    def loss(self, ŷ: float, y: float) -> float: return .5 * np.square(ŷ - y)
    def grad(self, ŷ: float, y: float) -> float: return ŷ - y

class LOGISTIC(Loss):
    def loss(self, ŷ: float, y: float) -> float:
        # clip values using eps=1e-15 (sklearn)
        ŷ = np.array(ŷ).clip(min=1e-15, max=1 - 1e-15)
        y = np.array(y)
        loss = -y * np.log(ŷ) - (1 - y) * np.log(1 - ŷ)
        return loss.mean()
    def grad(self, ŷ: float, y: float) -> float:
        return ŷ - y