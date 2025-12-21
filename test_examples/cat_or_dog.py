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


def get_images(img_size=128, max_photos=500):
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
    model = Model(name="cat_or_dog_vgg")

    X, y = get_images(img_size=128, max_photos=1000)
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

    tmp = model(X[0:1])

    model.add_layer(FlatteningLayer())
    model.add_layer(DenseLayer(tmp.size, 128, name="fc1"))
    model.add_layer(ReLU())

    model.add_layer(DenseLayer(128, 1, name="fc2"))
    model.add_layer(Sigmoid())

    model.set_loss(BinaryCrossEntropy())
    model.set_optimizer(RMSProp(2e-4, 0.98))

    model.fit(Dataset(train_X, train_y), batch_size=16, max_epochs=7, print_every=1, metrics=[BinaryAccuracy()])

    model.evaluate(Dataset(test_X, test_y), metrics=[BinaryAccuracy()])

    """

    This model was trained using a VGGNet architecture on a dataset of 1,000 images resized to 128×128 pixels.
    Training was performed with a batch size of 16 over 7 epochs and took 6 hours, 12 minutes, and 17 seconds
    (22,337 seconds) in total. The final model achieved an accuracy of 0.7 on the test set.

    Training:	Epoch: 1, loss: 0.7105776146361638.
    Metric: Training accuracy value: 0.5406
    --------------------------------------------------

    Training:	Epoch: 2, loss: 0.651739532191784.
    Metric: Training accuracy value: 0.6056
    --------------------------------------------------

    Training:	Epoch: 3, loss: 0.6165487161801664.
    Metric: Training accuracy value: 0.6575
    --------------------------------------------------

    Training:	Epoch: 4, loss: 0.5758527832430931.
    Metric: Training accuracy value: 0.7012
    --------------------------------------------------

    Training:	Epoch: 5, loss: 0.5430741257798565.
    Metric: Training accuracy value: 0.7356
    --------------------------------------------------

    Training:	Epoch: 6, loss: 0.5119452749895291.
    Metric: Training accuracy value: 0.7556
    --------------------------------------------------

    Training:	Epoch: 7, loss: 0.4713852604165614.
    Metric: Training accuracy value: 0.7719
    --------------------------------------------------    

    Parameters saved at saved_models/params_cat_or_dog_vgg.pickle
    Model saved at saved_models/model_cat_or_dog_vgg.pickle
    Test set loss = 0.5896997355367971
    Metric: accuracy, value: 0.7
    """