from typing import Dict

from backend.backend import xp, address
from optimizers.abstract_optimizer import Optimizer


class Adagrad(Optimizer):
    """
    Adagrad’s main idea is to adapt the learning rate per parameter: parameters that change rarely get larger updates, while
    those that change often get smaller ones. This is achieved by maintaining a diagonal matrix G whose entries accumulate
    squared gradients.

    At step t:
    gₜ = ∇J(wₜ),
    Gᵢᵢ(t) = Gᵢᵢ(t−1) + gₜ,ᵢ²,
    wₜ₊₁,ᵢ = wₜ,ᵢ − α gₜ,ᵢ / sqrt(Gᵢᵢ + eps).

    Larger values of Gᵢᵢ reduce the influence of gradient component gᵢ, meaning frequently updated parameters get smaller
    steps. eps (typically 1e−8) ensures numerical stability. The drawback is that G grows indefinitely, eventually making
    learning extremely slow. In practice, G can be stored as a vector with the same shape as the parameters.
    """

    def __init__(self, lr: float = 0.01):
        super().__init__(lr)
        self.g_sq: Dict[int, xp.ndarray] = {}
        self.eps = 1e-8

    def update_parameters(self, params: xp.ndarray, grad: xp.ndarray) -> xp.ndarray:
        a = address(params)

        if a not in self.g_sq:
            self.g_sq[a] = xp.zeros_like(grad)

        g_sq = self.g_sq[a]
        g_sq += grad*grad
        self.g_sq[a] = g_sq
        params -= self.lr * grad / xp.sqrt(g_sq + self.eps)

        return params
