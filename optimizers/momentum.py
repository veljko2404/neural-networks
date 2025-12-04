from typing import Dict

from optimizers.abstract_optimizer import Optimizer
from backend.backend import xp, address


class Momentum(Optimizer):
    """
    A common issue in gradient descent is oscillation near a local minimum. Momentum reduces oscillation by updating:

    vₜ = βvₜ₋₁ + α∇J(wₜ),
    wₜ₊₁ = wₜ − vₜ,

    with β typically 0.9. Opposing gradients partially cancel out, while consistent directions accumulate speed, helping the
    optimizer move faster across flat regions (plateaus).

    NESTEROV VARIANT
    Momentum behaves like a rolling ball that needs time to slow down or change direction. Nesterov momentum computes the
    gradient not at wₜ but at the “look-ahead” position wₜ − βvₜ₋₁, allowing quicker reaction when the direction should
    change:

    vₜ = βvₜ₋₁ + α∇J(wₜ − βvₜ₋₁),
    wₜ₊₁ = wₜ − vₜ.

    Dozat's modification for use with Adam approximates Nesterov momentum as:

    gₜ = ∇J(wₜ),
    vₜ = βvₜ₋₁ + αgₜ,
    wₜ₊₁ = wₜ − (βvₜ + αgₜ).
    """

    def __init__(self, lr: float = 0.005, beta: float = 0.9, nesterov: bool = False):
        super().__init__(lr)
        self.beta = beta
        self.v: Dict[int, xp.ndarray] = {}
        self.nesterov = nesterov  # s obzirom na sličnosti, ista klasa može vršiti i ažuriranje
        # u skladu sa Nesterovom modifikacijom momentuma.

    def update_parameters(self, params: xp.ndarray, grad: xp.ndarray) -> xp.ndarray:
        if address(params) not in self.v:
            self.v[address(params)] = xp.zeros_like(grad)

        v = self.v[address(params)]
        v = self.beta * v + self.lr * grad
        self.v[address(params)] = v

        if self.nesterov:
            params -= (self.beta * v + self.lr * grad)
        else:
            params -= v

        return params
