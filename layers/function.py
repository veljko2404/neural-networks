from abc import ABC, abstractmethod
from backend.backend import np

class Function(ABC):
    def __init__(self, name: str = "unnamed"):
        self._training = True
        self._name = name
        self._inputs: np.ndarray = None
        
    @abstractmethod
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        pass

    @property
    def training(self) -> bool:
        return self._training

    @training.setter
    def training(self, val: bool):
        self._training = val

    @property
    def parameters(self) -> tuple:
        return tuple()

    @parameters.setter
    def parameters(self, params: tuple):
        pass
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._pre_fw(inputs)

        return self(inputs)

    def _pre_fw(self, inputs: np.ndarray):
        if self._training:
            self._inputs = inputs

    @abstractmethod
    def backward(self, dEdO: np.ndarray) -> np.ndarray:
        pass