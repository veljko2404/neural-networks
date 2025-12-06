from loss_functions.cross_entropy import CrossEntropy
from metrics.metrics import Accuracy
from models.feedforward_nn import Model
import pandas as pd
from backend.backend import xp

from layers.activation_functions.relu import ReLU
from layers.dense_layer import DenseLayer
from optimizers.adam import Adam

def test_mnist():
    data = pd.read_csv('data/mnist.csv')
    data = xp.array(data)
    n, m = data.shape
    xp.random.shuffle(data)

    data_test = data[0:1000]
    test_y = data_test[:, 0].reshape(-1)
    test_X = data_test[:, 1:] / 255

    data_train = data[1000:n]
    train_y = data_train[:, 0].reshape(-1)
    train_X = data_train[:, 1:] / 255

    model = Model(name="mnist")
    model.add_layer(DenseLayer(train_X.shape[1], 32, name="DenseLayer1"))
    model.add_layer(ReLU())
    model.add_layer(DenseLayer(32, 16, name="DenseLayer2"))
    model.add_layer(ReLU())
    model.add_layer(DenseLayer(16, 10, name="DenseLayer3"))

    model.set_loss(CrossEntropy(from_logits=True, one_hot=False))
    model.set_optimizer(Adam(lr=0.005))
    acc = Accuracy(one_hot=False)

    model.fit((train_X,train_y), print_every=10, batch_size=32, max_epochs=70, metrics=[acc])
    acc_test = Accuracy(one_hot=False)
    model.evaluate((test_X, test_y), metrics=[acc_test])

    """
    Training time = 43.618773221969604 seconds
    Parameters saved at saved_models/mnist.pickle
    Test set loss = 0.3423822664738604
    Metric: accuracy, value: 0.965
    """

