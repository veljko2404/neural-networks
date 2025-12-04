from data_scalers.scalers import *
from models.feedforward_nn import Model
import pandas as pd
from backend.backend import xp

from layers.activation_functions.relu import ReLU
from layers.dense_layer import DenseLayer
from loss_functions.mse import MSE
from optimizers.sgd import SGD


def test_regression():
    file = pd.read_csv('data/housing.csv')
    """
    The file contains data on Boston houses and their prices. The task is to train a network to predict a house’s price
    based on its features.
    """
    data = xp.array(file.values)
    xp.random.shuffle(data)

    N = len(data)
    training_data_ratio = 0.8
    training_data_size = int(N * training_data_ratio)
    training_set = data[:training_data_size]
    test_set = data[training_data_size:]

    train_X, train_y = training_set[:, :-1], training_set[:, -1]
    test_X, test_y = test_set[:, :-1], test_set[:, -1]

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaler_x.adapt(train_X)
    scaler_y.adapt(train_y)

    train_X = scaler_x.transform(train_X)
    test_X = scaler_x.transform(test_X)
    train_y = scaler_y.transform(train_y)
    test_y = scaler_y.transform(test_y)

    model = Model(name="regression_1")
    model.add_layer(DenseLayer(train_X.shape[1], 16, name='Dense layer 1'))
    model.add_layer(ReLU())
    model.add_layer(DenseLayer(16, 32, name='Dense layer 2'))
    model.add_layer(ReLU())
    model.add_layer(DenseLayer(32, 1, name='Dense layer 3'))

    model.set_loss(MSE())
    model.set_optimizer(SGD(lr=0.1))
    model.fit((train_X, train_y), print_every=1, batch_size=64, max_epochs=200)

    # model.load_params("regression_1.pickle")
    model.evaluate((test_X, test_y))

    i = 7
    prediction = model(test_X[i-1:i])
    data = scaler_x.inverse(test_X[i - 1:i])
    prediction = scaler_y.inverse(prediction)
    y = scaler_y.inverse(test_y[i - 1:i])
    print("Inference: \n\tdata:{}\n\tprediction:{}\n\tlabel:{}".format(data, prediction, y))
