from typing import Tuple

from backend.backend import xp
from loss_functions.abstract_loss_function import LossFunction


class DKLStandardNormal(LossFunction):
    """
    A special case of KL divergence where we compute Dkl between a Gaussian q with parameters μq and exp(γ), and a Gaussian p
    with parameters μp = 0 and variance 1.
    """
    def __init__(self):
        super().__init__('KL Divergence')

    def __call__(self, mu: xp.ndarray, gamma: xp.ndarray) -> xp.ndarray:
        return -0.5 * xp.sum(1 + gamma - mu ** 2 - xp.exp(gamma)) / gamma.shape[0]

    def backward(self, mu: xp.ndarray, gamma: xp.ndarray) -> Tuple[xp.ndarray, xp.ndarray]:
        """
        Here we compute the partial derivatives of DKL with respect to μ and γ. We must also ensure that these derivatives are
        placed in the correct positions, since during backpropagation they are concatenated into a single longer vector, just
        like the inputs.
        """
        nb = mu.shape[0]
        return mu / nb, 0.5 * (xp.exp(gamma) - 1) / nb
