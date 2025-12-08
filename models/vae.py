from typing import Tuple

from backend.backend import xp
from models.feedforward_nn import Model
from loss_functions.abstract_loss_function import LossFunction
from loss_functions.kl_divergence import DKLStandardNormal
from optimizers.abstract_optimizer import Optimizer


class VAE(Model):
    """
    Given the functionality a variational autoencoder must support, the class will inherit from Network.
    The model consists of an encoder and a decoder. The encoder generates vectors mu and gamma from the input,
    which are then used to form the latent variable, z = mu + sqrt(exp(gamma)) * eps, where eps ~ N(0, 1).
    The encoder’s loss function is the KL divergence between N(mu, exp(gamma)) and N(0, 1).
    The decoder receives the latent variable z and aims to produce an output as similar as possible to the encoder’s input.
    Formally this corresponds to maximizing likelihood, leading the loss to be some form of entropy or MSE, depending on the assumed distribution p(X|z).

    During backpropagation, the decoder returns dE2/dz. Since the encoder’s final layer outputs mu and gamma,
    its parameters affect both E1 and E2. Therefore, the partial derivatives
    dE2/dmu = dz/dmu * dE2/dz  and  dE2/dgamma = dz/dgamma * dE2/dz  must be included.

    A practical challenge is that the Network class does not support branching—only sequential models with single-input/single-output loss functions.
    To avoid modifying the code, we use a simple trick: the encoder generates mu and gamma with a single Dense layer that outputs a vector of length 2D,
    where D is the latent dimension. The first half represents mu, the second half gamma.

    The KL loss receives this 2D vector, computes E1, and returns partial derivatives dE1/dmu and dE1/dgamma. These derivatives
    are then added to the corresponding derivatives from the decoder, producing a single gradient vector of size 2D. That combined
    vector is fed back through the encoder during backpropagation. The rest of the network behaves as usual.
    """

    class VAELoss(LossFunction):

        def __init__(self, decoder_loss: LossFunction):
            super().__init__("VAE loss")
            if decoder_loss is None:
                raise Exception("Decoder loss function cannot be None!")
            self.encoder_loss = DKLStandardNormal()
            self.decoder_loss = decoder_loss
            self.mu: xp.ndarray = None
            self.gamma: xp.ndarray = None

        def __call__(self, y: xp.ndarray, t: xp.ndarray) -> float:
            el = self.encoder_loss(self.mu, self.gamma)
            dl = self.decoder_loss(y, t)

            return el + dl

        def backward(self, y: xp.ndarray, t: xp.ndarray) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
            d_mu, d_gamma = self.encoder_loss.backward(self.mu, self.gamma)
            d_x = self.decoder_loss.backward(y, t)
            return d_mu, d_gamma, d_x

    def __init__(self, encoder: Model, decoder: Model, m: int, name="VAE"):
        super().__init__(name=name)

        self.encoder: Model = encoder
        self.decoder: Model = decoder

        self._loss = VAE.VAELoss(self.decoder._loss)

        self.m: int = m
        self.z: xp.ndarray = None
        self.eps: xp.ndarray = None
        self.gamma: xp.ndarray = None

    def set_optimizer(self, optimizer: Optimizer, force: bool = False):
        self.encoder.set_optimizer(optimizer)
        self.decoder.set_optimizer(optimizer)

    def update_parameters(self):
        self.encoder.update_parameters()
        self.decoder.update_parameters()

    @property
    def parameters(self) -> list:
        return [self.encoder.parameters,
                self.decoder.parameters]

    @parameters.setter
    def parameters(self, val: tuple):
        self.encoder.parameters, self.decoder.parameters = val

    def _get_z(self, mu_gamma: xp.ndarray) -> xp.ndarray:
        Nb = mu_gamma.shape[0]
        eps = xp.random.normal(size=(Nb, self.m))

        mu = mu_gamma[:, :self.m]
        gamma = mu_gamma[:, self.m:]

        self._loss.mu = mu
        self._loss.gamma = gamma

        z = eps * xp.exp(gamma * 0.5) + mu

        if self.training:
            self.eps = eps
            self.z = z
            self.gamma = gamma

        return z

    def __call__(self, inputs: xp.ndarray) -> xp.ndarray:
        tmp = self.encoder(inputs)
        self.z = self._get_z(tmp)
        return self.decoder(self.z)

    def backward(self, dE: xp.ndarray) -> xp.ndarray:
        d_mu, d_gamma, dx = dE

        dz = self.decoder.backward(dx)
        d_mu += dz
        d_gamma += dz * 0.5 * xp.exp(0.5 * self.gamma) * self.eps
        dmuGamma = xp.hstack((d_mu, d_gamma))

        return self.encoder.backward(dmuGamma)

    def generate_new_samples(self, num_of_samples: int = 1, samples_like: xp.ndarray = None) -> xp.ndarray:
        """
        The goal of a generative autoencoder is to produce new samples. These samples can be generated
        either based on existing examples or entirely at random.

        If we generate from existing samples, we repeat the same steps as in the forward pass.
        If we generate randomly, we simply feed the decoder with z values drawn from a standard normal distribution.
        """
        if samples_like is not None:
            return self(samples_like)

        z = xp.random.normal(size=(num_of_samples, self.m))
        return self.decoder(z)

    @property
    def training(self) -> bool:
        return self._training

    @training.setter
    def training(self, val: bool):
        self._training = val
        self.encoder.training = val
        self.decoder.training = val

