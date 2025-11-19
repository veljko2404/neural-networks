from backend.backend import np

from optimizers.abstract_optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, lr: float = 0.01):
        super().__init__(lr)

    def update_parameters(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        params -= self.lr * grad
        return params