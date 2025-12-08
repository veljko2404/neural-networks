from matplotlib import pyplot as plt
from matplotlib.pyplot import tight_layout

from backend.backend import xp
from data_scalers.scalers import MinMaxScaler
from layers.activation_functions.leaky_relu import LeakyReLU
from models.feedforward_nn import Model
from models.gan import GAN
from layers.activation_functions.relu import ReLU
from layers.activation_functions.sigmoid import Sigmoid
from layers.dense_layer import DenseLayer
from loss_functions.binary_cross_entropy import BinaryCrossEntropy
from metrics.metrics import BinaryAccuracy
from optimizers.rmsprop import RMSProp
from utils.utils import get_mnist_data


def test_GAN():
    X, y = get_mnist_data(flat_images=True)

    scaler = MinMaxScaler()
    X = scaler.transform(X)

    z_len = 16
    discr_dense_l = 64
    gen_dense_l = 128
    """
    We create two networks: a generator and a discriminator. Similar to the variational autoencoder, 
    we must ensure that the output and input dimensions  of the two networks match. The input to the generator 
    is random noise, and its output must have the same dimensions as the samples from the training dataset.
    The input to the discriminator must match the generator’s output dimensions (which naturally aligns with the 
    dimensions of the training data).
    The discriminator outputs the probability that a given sample comes from the real, original data distribution.
    """
    discr = Model(name="discriminator")
    discr.add_layer(DenseLayer(X.shape[-1], discr_dense_l))
    discr.add_layer(LeakyReLU())
    discr.add_layer(DenseLayer(discr_dense_l, 1))
    discr.add_layer(Sigmoid())
    discr.set_loss(BinaryCrossEntropy(from_logits=False))

    gen = Model(name="generator")
    gen.add_layer(DenseLayer(z_len, gen_dense_l))
    gen.add_layer(ReLU())
    gen.add_layer(DenseLayer(gen_dense_l, X.shape[-1]))
    gen.add_layer(ReLU())

    gan = GAN(gen, discr, z_len, 1)
    gan.set_optimizer(RMSProp())
    gan.fit((X, X), print_every=1, batch_size=100, max_epochs=30, metrics=[BinaryAccuracy(from_logits=False)])

    num_of_test_samples = 10
    s = gan.generate_new_samples(num_of_samples=num_of_test_samples)

    im_dim = int(xp.sqrt(X.shape[1]))
    s = s.reshape((-1, im_dim, im_dim))

    # if "cupy" in str(xp._version):
    #     s = xp.asnumpy(s)

    nrows = 1
    fig, axs = plt.subplots(nrows=nrows, ncols=num_of_test_samples, figsize=(16, 4), subplot_kw={'xticks': [], 'yticks': []})
    fig.suptitle('GAN test_examples')

    images = []
    for j in range(num_of_test_samples):
        images.append(axs[j].imshow(s[j], interpolation='nearest'))
        axs[j].label_outer()

    tight_layout()
    plt.show()

    """
    Training:	Epoch: 30, loss: 0.6909767768820936.
    Metric: Training accuracy value: 0.5362
    
    Training time = 137.56897854804993 seconds
    Parameters saved at saved_models/dnn_model.pickle
    
    Generated images saved to saved_model/gan_results_30_epochs.png
    
    Discriminator accuracy oscillates around 50–60%.
    This means it’s unsure whether a sample is real or fake — which is ideal.
    If accuracy is close to 100%, the generator is weak and learns nothing.
    If accuracy is close to 0%, the generator is too strong and the discriminator becomes useless.
    """