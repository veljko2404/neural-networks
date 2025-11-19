from abc import ABC, abstractmethod

from backend.backend import np
from layers.activation_functions.sigmoid import Sigmoid
from utils.utils import to_one_hot


class Metric(ABC):
    def __init__(self, _name: str):
        self.values_per_epoch = []
        self.name = _name

    @abstractmethod
    def calculate(self, y: np.ndarray, t: np.ndarray) -> float:
        pass

    @abstractmethod
    def calculate_for_epoch(self) -> float:
        pass

    def last_epoch_value(self) -> float:
        return np.round(self.values_per_epoch[-1], 4)  # radi lepšeg prikaza zaokružujemo rezultat na 4 decimale


class BinaryAccuracy(Metric):
    def __init__(self, from_logits: bool = False):
        super().__init__("accuracy")
        self.from_logits = from_logits
        self._count = 0
        self._sum = 0

    def calculate_for_epoch(self) -> float:
        a = self._sum / self._count
        self._sum = 0
        self._count = 0
        self.values_per_epoch.append(a)
        return a

    def calculate(self, y: np.ndarray, t: np.ndarray) -> float:
        if self.from_logits:
            y = Sigmoid()(y)
        a = np.sum(1 - np.abs(np.round(y) - t))

        self._sum += a
        self._count += y.size

        return float(a / y.size)


class Accuracy(Metric):
    def __init__(self, one_hot: bool = True):
        super().__init__("accuracy")
        self._count = 0
        self._sum = 0
        self.one_hot = one_hot

    def calculate_for_epoch(self) -> float:
        a = self._sum / self._count
        self._sum = 0
        self._count = 0
        self.values_per_epoch.append(a)
        return a

    def calculate(self, y: np.ndarray, t: np.ndarray) -> float:
        if not self.one_hot:
            t = to_one_hot(t, y.shape[-1])

        tmp1 = np.argmax(y, axis=-1)
        tmp2 = np.argmax(t, axis=-1)
        incorrect_classified = np.count_nonzero(tmp1 - tmp2)
        num_of_samples = y.size / y.shape[-1]
        a = (num_of_samples - incorrect_classified)

        self._sum += a
        self._count += num_of_samples

        return float(a / num_of_samples)
