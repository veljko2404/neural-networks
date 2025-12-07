from enum import Enum

from backend.backend import xp
from sklearn.datasets import load_iris, load_wine

from data_scalers.scalers import MinMaxScaler
from layers.activation_functions.softmax import Softmax
from models.feedforward_nn import Model
from layers.activation_functions.leaky_relu import LeakyReLU
from layers.dense_layer import DenseLayer
from layers.normalization.layer_normalization import LayerNormalization
from loss_functions.cross_entropy import CrossEntropy
from metrics.metrics import Accuracy
from optimizers.adam import Adam
from utils.dataset import Dataset


class UseDataset(Enum):
    WINE = 2
    IRIS = 3


def test_multiclass_classification(use_dataset: UseDataset = UseDataset.WINE):
    max_epochs = 150
    if use_dataset == UseDataset.IRIS:
        data = load_iris()
        # The Iris dataset contains 150 samples with four flower measurements (sepal length, sepal width,
        # petal length, petal width) used to classify three species: setosa, versicolor, and virginica.
    else:
        data = load_wine()
        # The Wine dataset contains 178 samples from three cultivars, each described by 13 input
        # features and classified into 3 output classes.

    X = xp.array(data['data'])
    y = xp.array(data['target'])
    print(data)

    xp.random.seed(123)
    xp.random.shuffle(X)
    xp.random.seed(123)
    xp.random.shuffle(y)

    num_of_classes = int(xp.max(y) + 1)
    y_one_hot = xp.zeros((len(y), num_of_classes), dtype=float)
    for i in range(len(y)):
        y_one_hot[i, y[i]] = 1

    N = len(X)
    training_data_ratio = 0.8
    m = int(N * training_data_ratio)

    train_X, test_X = X[: m], X[m: ]
    train_y, test_y = y_one_hot[: m], y_one_hot[m: ]

    scaler_x = MinMaxScaler()
    scaler_x.adapt(train_X)

    train_X = scaler_x.transform(train_X)
    test_X = scaler_x.transform(test_X)

    test_data = Dataset(test_X, test_y, batch_size=16)
    train_data = Dataset(train_X, train_y)

    normalization = LayerNormalization
    # normalization = BatchNormalization
    model = Model(name="multiclass_" + str(use_dataset.name))
    model.add_layer(DenseLayer(train_X.shape[1], 64, name='Dense layer 1'))
    # model.add_layer(normalization())
    model.add_layer(LeakyReLU())
    model.add_layer(DenseLayer(64, 32, name='Dense layer 2'))
    # model.add_layer(normalization())
    model.add_layer(LeakyReLU())
    model.add_layer(DenseLayer(32, num_of_classes, name='Dense layer 3'))

    model.add_layer(Softmax())
    model.set_loss(CrossEntropy(from_logits=False))
    # or
    # model.set_loss(CrossEntropy(from_logits=True, one_hot=True))

    model.set_optimizer(Adam(nesterov=True))
    model.fit(train_data, print_every=20, batch_size=64, max_epochs=max_epochs, metrics=[Accuracy(one_hot=True)])
    model.evaluate(test_data, metrics=[Accuracy(one_hot=True)])

    """
    WINE
    Training time = 0.2812039852142334 seconds
    Parameters saved at saved_models/multiclass_WINE.pickle
    Test set loss = 0.03490220510466558
    Metric: accuracy, value: 0.9722
    
    IRIS
    Training time = 0.24997806549072266 seconds
    Parameters saved at saved_models/multiclass_IRIS.pickle
    Test set loss = 0.11031155345444552
    Metric: accuracy, value: 0.9333
    """