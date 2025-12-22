from backend.backend import xp
from layers.function import Function


class Dropout(Function):
    def __init__(self, p: float = 0.5, name: str = "Dropout"):
        super().__init__(name)
        assert 0.0 <= p < 1.0
        self.p = p
        self.mask = None

    def __call__(self, inputs: xp.ndarray) -> xp.ndarray:
        self._inputs = inputs

        if not self.training or self.p == 0.0:
            return inputs

        self.mask = (xp.random.rand(*inputs.shape) > self.p) / (1.0 - self.p)
        return inputs * self.mask

    def backward(self, dEdO: xp.ndarray) -> xp.ndarray:
        if not self.training or self.p == 0.0:
            return dEdO

        return dEdO * self.mask
