from test_examples.basic_rnn import test_basic_RNN
from test_examples.binary_classification import test_binary_classification
from test_examples.btc_prices_lstm import btc_price_lstm
from test_examples.cat_or_dog import cat_or_dog
from test_examples.convolution_2D import test_CNN
from test_examples.lstm_and_gru import test_LSTM_GRU
from test_examples.multiclass_classification import test_multiclass_classification, UseDataset
from test_examples.regression import test_regression
from test_examples.test_gan import test_GAN
from test_examples.test_mnist import test_mnist
from test_examples.test_vae import test_VAE

if __name__ == '__main__':
    # test_regression()
    # test_mnist()
    # test_binary_classification()
    # test_multiclass_classification(UseDataset.IRIS)
    # test_basic_RNN()
    # test_LSTM_GRU()
    # test_CNN(True)
    # test_VAE()
    # test_GAN()
    cat_or_dog()
    # btc_price_lstm()
    pass