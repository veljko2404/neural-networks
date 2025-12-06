from abc import ABC, abstractmethod

from backend.backend import xp
from layers.activation_functions.sigmoid import Sigmoid
from utils.utils import to_one_hot


class Metric(ABC):
    """
    Different metrics are useful for evaluating training results and tracking progress. The key method is calculate_metric,
    which computes the metric for a given batch. Since we also want to track metrics across an entire epoch, the class
    provides calculate_for_epoch, which accumulates results over multiple batches.

    It is also helpful to inspect how a metric evolves over time, for example by plotting it. For this reason, the metric
    value for each epoch is stored.
    """

    def __init__(self, _name: str):
        self.values_per_epoch = []
        self.name = _name

    @abstractmethod
    def calculate(self, y: xp.ndarray, t: xp.ndarray) -> float:
        pass

    @abstractmethod
    def calculate_for_epoch(self) -> float:
        pass

    def last_epoch_value(self) -> float:
        return xp.round(self.values_per_epoch[-1], 4)  # round to 4 decimals


class BinaryAccuracy(Metric):
    """
    Accuracy is the ratio of correctly classified samples to the total number of samples. In classification tasks it is the
    metric we ultimately care about. Although minimizing the loss typically leads to higher accuracy, accuracy is far more
    readable during training—for example, 0.971 is more informative than a loss value like 0.00051.
    """

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

    def calculate(self, y: xp.ndarray, t: xp.ndarray) -> float:
        """
        Here we compute accuracy based on network_output and target_value. Binary and multiclass classification must be handled
        separately because their target representations differ.

        In binary classification each sample belongs to exactly one of two classes, labeled 0 or 1. Thus target_value is an
        Nb × 1 vector of zeros and ones, and the network output (after a sigmoid) is also Nb × 1, representing the probability
        that a sample belongs to class 1. We classify outputs > 0.5 as class 1 and ≤ 0.5 as class 0. A prediction is correct when
        abs(round(output) − target_value) = 0.

        Multiclass (non-multilabel) classification also assigns exactly one label per sample, but uses a different target format.
        """

        if self.from_logits:
            y = Sigmoid()(y)
        a = xp.sum(1 - xp.abs(xp.round(y) - t))

        self._sum += a
        self._count += y.size

        return float(a / y.size)


class Accuracy(Metric):
    """
    Accuracy is the ratio of correctly classified samples to the total number of samples. In classification tasks it is the
    metric we ultimately care about. Although reducing the loss usually increases accuracy, accuracy is far more readable
    during training—for example, 0.982 is much more meaningful than a loss of 0.000342.

    In multiclass classification, target values are represented differently. With k classes, a sample belonging to class 5 is
    not encoded as the number 5, but as a k-dimensional one-hot vector: the 5th position is 1 and all others are 0. This
    tells the network to output probability 1 for the correct class and 0 for all others.

    The network output uses a softmax activation, producing a probability distribution over classes. A sample is assigned to
    the class with the highest predicted probability. It is correctly classified when argmax(output) = argmax(target). When
    working with batches, argmax must be computed along the class dimension.
    """

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

    def calculate(self, y: xp.ndarray, t: xp.ndarray) -> float:
        if not self.one_hot:
            t = to_one_hot(t, y.shape[-1])

        tmp1 = xp.argmax(y, axis=-1)
        tmp2 = xp.argmax(t, axis=-1)
        incorrect_classified = xp.count_nonzero(tmp1 - tmp2)
        num_of_samples = y.size / y.shape[-1]
        a = (num_of_samples - incorrect_classified)

        self._sum += a
        self._count += num_of_samples

        return float(a / num_of_samples)

class MSEMetric(Metric):
    def __init__(self):
        super().__init__("MSE")
        self._sum = 0
        self._count = 0

    def calculate(self, y, t):
        mse = xp.mean((y - t)**2)
        self._sum += mse
        self._count += 1
        return float(mse)

    def calculate_for_epoch(self):
        value = self._sum / self._count
        self.values_per_epoch.append(value)
        self._sum = 0
        self._count = 0
        return value