import pandas as pd

from data_scalers.scalers import *
from layers.activation_functions.sigmoid import Sigmoid
from models.feedforward_nn import Model
from layers.activation_functions.relu import ReLU
from layers.dense_layer import DenseLayer
from loss_functions.binary_cross_entropy import BinaryCrossEntropy
from metrics.metrics import BinaryAccuracy
from optimizers.adadelta import Adadelta
from optimizers.adagrad import Adagrad
from optimizers.adamax import AdaMax
from optimizers.sgd import SGD
from utils.dataset import Dataset


def test_binary_classification():
    dataset_name = "banknote_authentication"
    dataset_name = "heart"

    #file = pd.read_csv('data/' + dataset_name + '.csv')
    file = pd.read_csv('data/banknote_authentication.csv')

    # The file contains patient information and a target column indicating the class label.
    # Each sample belongs to one of two classes: 1 – high risk of heart attack, 0 – low risk.
    # Dataset source: https://www.kaggle.com/nareshbhat/health-care-data-set-on-heart-attack-possibility

    # Example of another dataset (banknote authentication):
    # Source: http://archive.ics.uci.edu/ml/datasets/banknote+authentication

    data = xp.array(file.values)
    xp.random.shuffle(data)
    N = len(data)

    training_data_ratio = 0.7
    training_data_size = int(N * training_data_ratio)

    y = data[:, -1]
    X = data[:, :-1]

    train_X, test_X = X[: training_data_size], X[training_data_size:]
    train_y, test_y = y[: training_data_size], y[training_data_size:]

    scaler_x = MinMaxScaler()

    scaler_x.adapt(train_X)

    train_X = scaler_x.transform(train_X)
    test_X = scaler_x.transform(test_X)

    m = len(test_X)//2
    val_X, test_X = test_X[:m], test_X[m:]
    val_y, test_y = test_y[:m], test_y[m:]

    test_data = Dataset(test_X, test_y, shuffle=False)
    val_data = Dataset(val_X, val_y, shuffle=False)
    train_data = Dataset(train_X, train_y)

    model = Model(name="bin_class_" + dataset_name)
    model.add_layer(DenseLayer(train_X.shape[1], 64, name='Dense layer 1'))
    model.add_layer(ReLU())
    model.add_layer(DenseLayer(64, 32, name='Dense layer 2'))
    model.add_layer(ReLU())
    model.add_layer(DenseLayer(32, 1, name='Dense layer 3'))
    from_logits = True

    # model.add_layer(Sigmoid())
    # from_logits = False

    loss = BinaryCrossEntropy(from_logits=from_logits)
    model.set_loss(loss)
    model.set_optimizer(Adadelta())

    model.fit(train_data, val_data, print_every=1, batch_size=32, max_epochs=50, metrics=[BinaryAccuracy(from_logits)])

    model.evaluate(test_data, metrics=[BinaryAccuracy(from_logits)])

    """
    HEART 
    Training time = 0.009022712707519531 seconds
    Parameters saved at saved_models/bin_class_heart.pickle
    Test set loss = 0.35119644770439135
    Metric: accuracy, value: 0.913
    
    Training time = 0.03310084342956543 seconds
    Parameters saved at saved_models/bin_class_heart.pickle
    Test set loss = 0.23189195297890325
    Metric: accuracy, value: 0.9126
    """