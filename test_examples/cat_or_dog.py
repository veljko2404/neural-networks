from layers.activation_functions.sigmoid import Sigmoid
from loss_functions.binary_cross_entropy import BinaryCrossEntropy
from models.feedforward_nn import Model
from layers.activation_functions.relu import ReLU
from layers.conv_layer.conv_layer_algorithms import *
from layers.conv_layer.convolution_layer import Convolution2D
from layers.conv_layer.pooling import Pooling
from layers.dense_layer import DenseLayer
from layers.flattening_layer import FlatteningLayer
from metrics.metrics import BinaryAccuracy
from optimizers.rmsprop import RMSProp
from utils.dataset import Dataset
import os
import cv2

def get_images(img_size = 128, max_photos = 1000):
    X = []
    y = []
    base_path = "data/PetImages"
    classes = {"Cat": 0, "Dog": 1}
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

def cat_or_dog():
    model = Model(name="cat_or_dog")

    X, y = get_images(img_size = 128, max_photos = 500)
    X = X.transpose(0, 3, 1, 2)

    idx = xp.random.permutation(len(X))
    X, y = X[idx], y[idx]

    split = int(0.8 * len(X))
    train_X, test_X = X[:split], X[split:]
    train_y, test_y = y[:split], y[split:]

    algorithm = Matmul

    model.add_layer(Convolution2D(3, 6, 5, padding=2, algorithm=algorithm()))
    model.add_layer(Pooling(6, kernel_size=2, stride=2))
    model.add_layer(ReLU())

    model.add_layer(Convolution2D(6, 16, 3, padding=1, algorithm=algorithm()))
    model.add_layer(Pooling(16, kernel_size=2, stride=2))
    model.add_layer(ReLU())

    tmp = model(X[0:1])

    model.add_layer(FlatteningLayer())
    model.add_layer(DenseLayer(tmp.size, 32, name='Dense layer 1'))
    model.add_layer(ReLU())

    model.add_layer(DenseLayer(32, 1, name='Dense layer 2'))
    model.add_layer(Sigmoid())
    model.set_loss(BinaryCrossEntropy())

    model.set_optimizer(RMSProp(2e-4, 0.98))

    model.fit(Dataset(train_X, train_y), print_every=1, batch_size=32, max_epochs=7, metrics=[BinaryAccuracy()])
    model.evaluate(Dataset(test_X, test_y), metrics=[BinaryAccuracy()])

    """
    Training:	Epoch: 1, loss: 0.9852502701681126.
    Metric: Training accuracy value: 0.5062
    --------------------------------------------------
    
    Training:	Epoch: 2, loss: 0.7493922622857921.
    Metric: Training accuracy value: 0.5538
    --------------------------------------------------
    
    Training:	Epoch: 3, loss: 0.660190242526714.
    Metric: Training accuracy value: 0.5875
    --------------------------------------------------
    
    Training:	Epoch: 4, loss: 0.6674418362996009.
    Metric: Training accuracy value: 0.6238
    --------------------------------------------------
    
    Training:	Epoch: 5, loss: 0.6005293695966163.
    Metric: Training accuracy value: 0.6738
    --------------------------------------------------
    
    Training:	Epoch: 6, loss: 0.5817361235875602.
    Metric: Training accuracy value: 0.7025
    --------------------------------------------------
    
    Training:	Epoch: 7, loss: 0.5746428384036584.
    Metric: Training accuracy value: 0.6888
    --------------------------------------------------
    
    Training time = 1028.6819405555725 seconds
    Parameters saved at saved_models/params_cat_or_dog.pickle
    Model saved at saved_models/model_cat_or_dog.pickle
    Test set loss = 0.6410190798913474
    Metric: accuracy, value: 0.61
    """