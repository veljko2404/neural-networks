from typing import List

from backend.backend import xp
from models.feedforward_nn import Model
from metrics.metrics import Metric
from optimizers.abstract_optimizer import Optimizer
from utils.dataset import Dataset


class GAN(Model):
    """
    Generative Adversarial Networks (GANs) must support training, forward propagation, backward propagation of
    partial derivatives, parameter updates, and so on. They require almost all functionalities of the Network
    class, which is why this class, like the VAE class, is derived from Network.

    We need two networks: a generator and a discriminator. The generator’s task is to take random noise as input and produce
    outputs that resemble samples from the training dataset. The discriminator’s task is to learn to distinguish between
    “real” samples (coming from the training distribution) and “fake” samples (coming from the generator).

    In implementation terms, the discriminator outputs the probability that its input is a real training sample rather than a
    generator-produced sample. Therefore, its output dimension is 1, and its loss function is binary cross-entropy.

    The generator outputs data with the same format as the training samples, and its input is random noise. The generator does not
    have its own loss function; instead, its output is fed into the discriminator, and through the discriminator’s gradients the generator is trained.

    GANs are not trained by updating both networks simultaneously. Instead, we train them alternately. When training the discriminator,
    we feed it examples from both classes: “real” and “fake”. Fake samples are obtained by passing random noise through the generator.
    Real samples are labeled as 1, fake samples as 0. To simplify implementation, the discriminator is trained using two
    separate batches—one with real data and one with generated data.

    When training the generator, the input is random noise, and the goal is for the generator to transform that noise into samples that
    match the format of the training data and are similar enough that the discriminator classifies them as real.
    From the generator’s optimization perspective, the target label for generated samples during its training phase is 1.
    """
    def __init__(self, generator: Model, discriminator: Model, z_size: int, k: int = 1):
        super().__init__()

        self.generator: Model = generator
        self.discriminator: Model = discriminator
        self.z_size = z_size
        self.k = k

        self.train_generator = True
        self.noise = False

    def set_optimizer(self, optimizer: Optimizer, force: bool = False):
        self.generator.set_optimizer(optimizer)
        self.discriminator.set_optimizer(optimizer)

    def _epoch(self, data: Dataset,
               metrics: List[Metric] = []) -> float:
        loss = 0.0
        batch_num = 0

        for batch_x, _ in data:

            batch_num += 1
            nb = len(batch_x)
            x_fake = self.generate_new_samples(nb)
            y_0 = xp.zeros((nb, 1), dtype=float)
            y_1 = xp.ones((nb, 1), dtype=float)
            x = xp.vstack((x_fake, batch_x))
            y = xp.vstack((y_0, y_1))

            output, _, l = self.discriminator._process_minibatch(x, y)
            self.discriminator.update_parameters()
            loss += l

            for m in metrics:
                m.calculate(output, y)

            if batch_num % self.k == 0:
                x_fake = self.generate_new_samples(nb)
                _, dEdG, _ = self.discriminator._process_minibatch(x_fake, y_1)
                self.generator.backward(dEdG)
                self.generator.update_parameters()

        loss /= batch_num
        for m in metrics:
            m.calculate_for_epoch()
        return loss

    def generate_new_samples(self, num_of_samples: int = 1) -> xp.ndarray:
        z = xp.random.normal(0, 1, (num_of_samples, self.z_size))
        return self.generator(z)

    @property
    def parameters(self) -> list:
        return [self.generator.parameters,
                self.discriminator.parameters]

    @parameters.setter
    def parameters(self, val: tuple):
        self.generator.parameters, self.discriminator.parameters = val

    @property
    def training(self) -> bool:
        return self._training

    @training.setter
    def training(self, val: bool):
        self._training = val
        self.generator.training = val
        self.discriminator.training = val
