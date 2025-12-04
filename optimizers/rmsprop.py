from typing import Dict

from optimizers.abstract_optimizer import Optimizer
from backend.backend import xp, address


class RMSProp(Optimizer):
    """
    RMSProp updates parameters as:

    xₜ₊₁ = xₜ − learning_rate * gradient / RMS(E[g]ₜ),

    where
    RMS(E[g]ₜ) = sqrt(E[g²]ₜ + eps),
    E[g²]ₜ = βE[g²]ₜ₋₁ + (1 − β) gradient²,
    and eps ≈ 1e−8.

    By accumulating squared gradients, RMSProp increases updates for parameters that change rarely and decreases them for
    those that change frequently.
    """

    def __init__(self, lr: float = 0.001, beta: float = 0.9):
        super().__init__(lr)
        self.beta = beta
        self.grad_sq: Dict[int, xp.ndarray] = {}
        self.eps = 1e-8

    def update_parameters(self, params: xp.ndarray, grad: xp.ndarray) -> xp.ndarray:
        if address(params) not in self.grad_sq:
            self.grad_sq[address(params)] = xp.zeros_like(grad)

        grad_sq = self.grad_sq[address(params)]
        grad_sq = self.beta * grad_sq + (1 - self.beta) * grad * grad
        self.grad_sq[address(params)] = grad_sq
        params -= self.lr * grad / xp.sqrt(grad_sq + self.eps)

        return params
