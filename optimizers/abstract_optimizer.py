from abc import ABC, abstractmethod
from backend.backend import np

class Optimizer(ABC):

    def __init__(self, lr: float):
        self.lr = lr

    @abstractmethod
    def update_parameters(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        pass
