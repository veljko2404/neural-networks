from models.feedforward_nn import Model
from layers.dense_layer import DenseLayer
from layers.recurrent_layers.gru import GRU
from layers.recurrent_layers.lstm import LSTM
from loss_functions.cross_entropy import CrossEntropy
from metrics.metrics import Accuracy
from optimizers.adam import Adam
from utils.dataset import Dataset
from utils.utils import generate_vocabulary, generate_sequences
from backend.backend import xp


def test_LSTM_GRU():
    model = Model(name="LSTM_GRU")

    content_str = open('data/dino.txt').read()
    seq_len = "auto"
    seq_delimiter = '\n'

    content_str = content_str.lower()
    print(content_str)
    char_to_int, int_to_char = generate_vocabulary(content_str, '/')

    X, y = generate_sequences(content_str, seq_length=seq_len, seq_delimiter=seq_delimiter, padding='/')

    model.add_layer(DenseLayer(X.shape[-1], 32))
    model.add_layer(GRU(32, 128))
    # model.add_layer(LSTM(32, 128))
    model.add_layer(DenseLayer(128, len(char_to_int)))

    model.set_loss(CrossEntropy())

    model.set_optimizer(Adam())
    batch_size = min(50, X.shape[0])
    model.fit(Dataset(X, y), print_every=1, batch_size=batch_size, max_epochs=50, metrics=[Accuracy()])
    xp.random.shuffle(X)

    for l in model._layers:
        if isinstance(l, LSTM) or isinstance(l, GRU):
            l.reset_state = False

    network_input = xp.zeros((batch_size, 1, len(char_to_int)))
    generated_text = [''] * batch_size
    print()

    warm_up = 3  # How many characters we feed into the network to let it compute and update the recurrent state.

    for t in range(warm_up):
        for i in range(batch_size):
            network_input[i, 0, :] = X[i, t, :]
            c = int_to_char[int(xp.argmax(network_input[i, 0, :]))]
            generated_text[i] += c
        network_output = model.forward(network_input)

    text_len = 50
    for t in range(text_len):
        for i in range(batch_size):
            # xp.random.choice() selects a random element from an array using the probabilities given by p.
            # Here we pass range(len(char_to_int)), i.e., 0, 1, 2, ..., len(char_to_int) - 1.
            # c_idx = int(xp.random.choice(range(len(char_to_int)), p=Softmax()(network_output[i, 0, :]), size=1))

            c_idx = int(xp.argmax(network_output[i, 0, :]))
            generated_text[i] += int_to_char[c_idx]  # We convert the sampled index into a character and append it to the generated text.
            network_input[i] = 0
            # We continue using network_input as the network’s input. Since the input is one-hot encoded,
            # we need to set a single 1 in the correct position.
            # The simplest approach is to reset all values to zero and then set the appropriate index to 1.
            network_input[i, 0, c_idx] = 1
        network_output = model.forward(network_input)

    for i in range(batch_size):
        print(generated_text[i])
        print('-----------------------')

    """
    GRU
    
    Training time = 0.2927229404449463 seconds
    Parameters saved at saved_models/LSTM_GRU.pickle
    
    lenouuuauas//////////////////////////////////////////
    -----------------------
    opeoauuuasu//////////////////////////////////////////
    -----------------------
    osauuar//////////////////////////////////////////////
    -----------------------
    paluuuuau////////////////////////////////////////////
    -----------------------
    sasuur///////////////////////////////////////////////
    -----------------------
    saurar///////////////////////////////////////////////
    -----------------------
    agisuuuar////////////////////////////////////////////
    -----------------------
    ausuuar//////////////////////////////////////////////
    -----------------------
    suuarrar/////////////////////////////////////////////
    -----------------------
    lurauuarsu///////////////////////////////////////////
    -----------------------
    spiouauuas///////////////////////////////////////////
    -----------------------
    droauasusu///////////////////////////////////////////
    -----------------------
    saurar///////////////////////////////////////////////
    -----------------------
    ediouauuas///////////////////////////////////////////
    -----------------------
    eyeoauuasu///////////////////////////////////////////
    -----------------------
    lenouuuauas//////////////////////////////////////////
    -----------------------
    """