from typing import Union, Dict

from optimizers.abstract_optimizer import Optimizer
from backend.backend import xp, address


class Adam(Optimizer):
    """
    Like Adadelta and RMSprop, Adam (Adaptive Moment Estimation) uses exponential moving averages of gradients and squared
    gradients. At step t:

    mₜ = β₁ mₜ₋₁ + (1 − β₁) gₜ,
    vₜ = β₂ vₜ₋₁ + (1 − β₂) gₜ².

    These estimates are biased toward zero at the beginning, so Adam applies bias correction:

    m̂ₜ = mₜ / (1 − β₁ᵗ),
    v̂ₜ = vₜ / (1 − β₂ᵗ).

    The parameter update is:

    wₜ₊₁ = wₜ − α m̂ₜ / (sqrt(v̂ₜ) + eps),

    where eps ≈ 1e−8 for numerical stability, and β₂ is typically 0.999.
    """

    def __init__(self, lr: float = 0.002, beta_1: float = 0.9, beta_2: float = 0.999, nesterov: bool = False):
        super().__init__(lr)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.history: Dict[int, Dict[str, Union[xp.ndarray, int]]] = {}
        self.eps = 1e-8
        self.nesterov = nesterov

    def update_parameters(self, params: xp.ndarray, grad: xp.ndarray) -> xp.ndarray:
        a = address(params)
        if a not in self.history:
            self.history[a] = {}
            self.history[a]["t"] = 0
            self.history[a]["v"] = xp.zeros_like(grad)
            self.history[a]["m"] = xp.zeros_like(grad)

        self.history[a]["t"] += 1
        self.history[a]["m"] = self.beta_1 * self.history[a]["m"] + (1 - self.beta_1) * grad
        self.history[a]["v"] = self.beta_2 * self.history[a]["v"] + (1 - self.beta_2) * grad * grad

        m_corr = self.history[a]["m"] / (1 - xp.power(self.beta_1, self.history[a]["t"]))
        v_corr = self.history[a]["v"] / (1 - xp.power(self.beta_2, self.history[a]["t"]))

        if self.nesterov:
            beta_t = xp.power(self.beta_1, self.history[a]["t"])
            params -= self.lr * (self.beta_1 * m_corr + (1-self.beta_1)*grad/(1 - beta_t)) / (xp.sqrt(v_corr) + self.eps)
        else:
            params -= self.lr * m_corr / (xp.sqrt(v_corr) + self.eps)
        return params
