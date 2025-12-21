from layers.activation_functions.softmax import Softmax
from loss_functions.cross_entropy import CrossEntropy
from models.feedforward_nn import Model
from layers.activation_functions.relu import ReLU
from layers.conv_layer.conv_layer_algorithms import *
from layers.conv_layer.convolution_layer import Convolution2D
from layers.conv_layer.pooling import Pooling
from layers.dense_layer import DenseLayer
from layers.flattening_layer import FlatteningLayer
from metrics.metrics import BinaryAccuracy, Accuracy
from optimizers.rmsprop import RMSProp
from utils.dataset import Dataset
import os
import cv2


def get_images(img_size=128, max_photos=500):
    X = []
    y = []
    base_path = "data/five cats photos"
    classes = {"caracal": 0, "cheetah": 1, "lions": 2, "puma": 3, "tiger": 4}
    for class_name, label in classes.items():
        folder = os.path.join(base_path, class_name)
        count = 0
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            if count >= max_photos:
                break

            img = cv2.imread(path)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))

            X.append(img)
            y.append(label)
            count += 1

    X = xp.array(X, dtype=xp.float32) / 255.0
    y = xp.array(y, dtype=xp.int64)

    return X, y

def five_cat_species():
    model = Model(name="five_cat_species")

    X, y = get_images(img_size=96, max_photos=200)
    X = X.transpose(0, 3, 1, 2)

    idx = xp.random.permutation(len(X))
    X, y = X[idx], y[idx]

    split = int(0.8 * len(X))
    train_X, test_X = X[:split], X[split:]
    train_y, test_y = y[:split], y[split:]

    algorithm = Matmul

    model.add_layer(Convolution2D(3, 16, 3, padding=1, algorithm=algorithm()))
    model.add_layer(ReLU())
    model.add_layer(Convolution2D(16, 16, 3, padding=1, algorithm=algorithm()))
    model.add_layer(ReLU())
    model.add_layer(Pooling(16, kernel_size=2, stride=2))

    model.add_layer(Convolution2D(16, 32, 3, padding=1, algorithm=algorithm()))
    model.add_layer(ReLU())
    model.add_layer(Convolution2D(32, 32, 3, padding=1, algorithm=algorithm()))
    model.add_layer(ReLU())
    model.add_layer(Pooling(32, kernel_size=2, stride=2))

    model.add_layer(Convolution2D(32, 64, 3, padding=1, algorithm=algorithm()))
    model.add_layer(ReLU())
    model.add_layer(Convolution2D(64, 64, 3, padding=1, algorithm=algorithm()))
    model.add_layer(ReLU())
    model.add_layer(Pooling(64, kernel_size=2, stride=2))

    model.add_layer(FlatteningLayer())
    tmp = model(X[0:1])

    model.add_layer(DenseLayer(tmp.shape[1], 128, name="fc1"))
    model.add_layer(ReLU())

    model.add_layer(DenseLayer(128, 5, name="fc2"))
    model.add_layer(Softmax())
    model.set_loss(CrossEntropy())
    model.set_optimizer(RMSProp(2e-4, 0.98))

    model.fit(Dataset(train_X, train_y), batch_size=8, max_epochs=10, print_every=1, metrics=[Accuracy()])

    model.evaluate(Dataset(test_X, test_y), metrics=[Accuracy()])
