from backend.backend import xp

from optimizers.abstract_optimizer import Optimizer


class SGD(Optimizer):
    """
    Basic gradient descent is the simplest optimization algorithm. Parameters are updated as
    x = x − lr * g,
    where lr is the learning rate and g is the gradient.
    """
    def __init__(self, lr: float = 0.01):
        super().__init__(lr)

    def update_parameters(self, params: xp.ndarray, grad: xp.ndarray) -> xp.ndarray:
        params -= self.lr * grad
        return params
