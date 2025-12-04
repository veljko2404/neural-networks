from typing import Dict

from backend.backend import xp, address
from optimizers.abstract_optimizer import Optimizer


class Adadelta(Optimizer):
    """
    Adadelta solves the training-stall problem seen in Adagrad. Instead of accumulating the sum of all past squared
    gradients, it maintains an exponential moving average E[g²]ₜ. This average is updated as:

    E[g²]ₜ = βE[g²]ₜ₋₁ + (1 − β)g²,

    where β (typically 0.9) controls how much past values influence the new estimate. We then define:

    RMS[g]ₜ = sqrt(E[g²]ₜ + eps),
    RMS[Δw]ₜ₋₁ = sqrt(E[Δw²]ₜ₋₁ + eps).

    The parameter update is:

    gₜ = ∇J(wₜ),
    Δwₜ = − RMS[Δw]ₜ₋₁ * gₜ / RMS[g]ₜ,
    E[Δw²]ₜ = βE[Δw²]ₜ₋₁ + (1 − β)Δw²ₜ,
    wₜ₊₁ = wₜ + Δwₜ.
    """

    def __init__(self, beta: float = 0.9):
        super().__init__(0)
        self.history: Dict[int, Dict[str, xp.ndarray]] = {}
        self.eps = 1e-6
        self.beta = beta

    def update_parameters(self, params: xp.ndarray, grad: xp.ndarray) -> xp.ndarray:
        a = address(params)
        if a not in self.history:
            self.history[a] = {}
            self.history[a]["E_g_sq"] = xp.zeros_like(grad)
            self.history[a]["E_update_sq"] = xp.zeros_like(grad)

        self.history[a]["E_g_sq"] = self.beta * self.history[a]["E_g_sq"] + (1 - self.beta) * grad * grad
        update = - xp.sqrt(self.history[a]["E_update_sq"] + self.eps) * grad / xp.sqrt(self.history[a]["E_g_sq"] + self.eps)
        self.history[a]["E_update_sq"] = self.beta * self.history[a]["E_update_sq"] + (1 - self.beta) * update * update
        params += update

        return params
