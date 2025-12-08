from matplotlib import pyplot as plt
from matplotlib.pyplot import tight_layout

from backend.backend import xp
from data_scalers.scalers import MinMaxScaler
from models.feedforward_nn import Model
from layers.activation_functions.relu import ReLU
from layers.activation_functions.sigmoid import Sigmoid
from layers.dense_layer import DenseLayer
from loss_functions.binary_cross_entropy import BinaryCrossEntropy
from optimizers.rmsprop import RMSProp
from utils.dataset import Dataset
from utils.utils import get_mnist_data
from models.vae import VAE


def test_VAE():
    X, y = get_mnist_data(flat_images=True)

    scaler = MinMaxScaler()
    X = scaler.transform(X)
    X = xp.where(X > 0.5, 1.0, 0.0)
    """
    We can reduce the images to strictly black and white — only black and only white pixels.
    During image reconstruction, the decision for each pixel is binary: should it be black or white,
    i.e., should its value be 1 or 0. In that case, the decoder should use binary cross-entropy as the loss function.
    """

    z_len = 16
    dense_layer_size = 128
    """
    We create two networks, an encoder and a decoder.
    We must pay attention to the output dimensions of the encoder and the input dimensions of the decoder.
    The encoder’s output should be the values gamma and mu.
    To reuse the existing architecture and classes without modifying anything, the encoder
    is implemented so that it returns an output of size 2 × len(z), where z is the latent random variable.
    We treat the first half of that output as mu, and the second half as gamma.
    What matters here is that, keeping this in mind, the encoder’s output dimension and the decoder’s input
    dimension must be set in a 2 : 1 ratio. We will not define a loss function for the encoder network.
    """

    encoder = Model()
    encoder.add_layer(DenseLayer(X.shape[-1], dense_layer_size))
    encoder.add_layer(ReLU())
    encoder.add_layer(DenseLayer(dense_layer_size, z_len * 2))

    decoder = Model()
    decoder.add_layer(DenseLayer(z_len, dense_layer_size))
    decoder.add_layer(ReLU())
    decoder.add_layer(DenseLayer(dense_layer_size, X.shape[-1]))
    decoder.add_layer(Sigmoid())
    """
    If we use MSE as the loss function for this problem, where target values are in [0, 1],
    we can use the sigmoid activation function. But it’s not required — the decoder can work without it.
    The decoder's output will be a 1D vector of size H*W (for a single example; with mini-batch training it will be Nb × H*W).
    That output only needs to be reshaped to obtain a 2D image.
    """

    # decoder.set_loss(MSE())
    decoder.set_loss(BinaryCrossEntropy(from_logits=False))
    vae = VAE(encoder, decoder, z_len)
    vae.set_optimizer(RMSProp())

    num_of_epochs = 30
    vae.fit(Dataset(X, X), print_every=1, batch_size=100, max_epochs=num_of_epochs)
    # vae.load_params("VAE.pickle")
    """
    After training is finished, we will take num_of_test_samples images from the training set and pass them through 
    the variational autoencoder. We will display the original samples and the reconstructed results on a plot.
    """
    num_of_test_samples = 10
    generate_random_samples = False
    # for k in range(5):
    # vae.decoder.add_layer(Sigmoid())
    xp.random.seed(11)
    xp.random.shuffle(X)
    if generate_random_samples:
        s = vae.generate_new_samples(num_of_samples=num_of_test_samples)
    else:
        s = vae.generate_new_samples(num_of_samples=num_of_test_samples, samples_like=X[:num_of_test_samples])
    im_dim = int(xp.sqrt(X.shape[1]))
    s = s.reshape((-1, im_dim, im_dim))

    nrows = 1 if generate_random_samples else 2
    fig, axs = plt.subplots(nrows=nrows, ncols=num_of_test_samples, figsize=(16, 4), subplot_kw={'xticks': [], 'yticks': []})
    # fig.suptitle('VAE test_examples, num of classes = {}, num of epochs = {}'.format(num_of_classes, num_of_epochs))

    # if "cupy" in str(xp._version):
    #     X = xp.asnumpy(X)
    #     s = xp.asnumpy(s)

    images = []
    for j in range(num_of_test_samples):
        if generate_random_samples:
            images.append(axs[j].imshow(s[j], interpolation='nearest'))
            axs[j].label_outer()

        else:
            images.append(axs[0, j].imshow(X[j].reshape(im_dim, im_dim), interpolation='nearest'))
            images.append(axs[1, j].imshow(s[j], interpolation='nearest'))
            axs[0, j].label_outer()
            axs[1, j].label_outer()

    tight_layout()
    plt.show()

    """
    Training time = 40.610310077667236 seconds
    Generated images are saved in saved_models/vae_results_10_epochs.png
                                  saved_models/vae_results_30_epochs.png
    """