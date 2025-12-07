from backend.backend import xp
from models.feedforward_nn import Model
from layers.activation_functions.relu import ReLU
from layers.dense_layer import DenseLayer
from layers.recurrent_layers.basic_recurrent import RecurrentLayer
from loss_functions.cross_entropy import CrossEntropy
from metrics.metrics import Accuracy
from optimizers.rmsprop import RMSProp
from utils.dataset import Dataset
from utils.utils import generate_vocabulary, generate_sequences


def test_basic_RNN():
    model = Model(name="VanillaRNN")

    content_str = open('data/usernames.txt').read()
    seq_len = "auto"
    seq_delimiter = '\n'

    """
    One of the simpler examples that can be learned fairly well within a reasonable amount of time is generating dinosaur names.
    Basic recurrent network with a single recurrent layer performed best for this task.
    After training, the model can produce generated strings that resemble real dinosaur names:
        eyeotoaeaar
        dromeaosssasas
        osarurs
        opeltauus
        sassus
        agilehsurauu
        sauruaur
        suurancsauar
        saururemeaurus
        edicyraur
        luraghitimau
        ausaueurorusaur
        lenguurusausus
        spingoaususaur
        palelounuesaus
        lengusfosaus
    """

    content_str = content_str.lower()
    # Dictionary objects below
    # One maps each character to an integer, and the other performs the inverse mapping.
    char_to_int, int_to_char = generate_vocabulary(content_str, '.')
    # generate_sequences returns the xp.ndarray arrays X and y used for training.
    # Characters in both arrays are one-hot encoded, and the arrays have shape Ns × T × D,
    # where Ns is the number of sequences of length T and D is the number of distinct characters.
    # Note that Ns × T ≤ N, where N is the total number of characters in the full text.
    X, y = generate_sequences(content_str, seq_length=seq_len, seq_delimiter=seq_delimiter, padding='/')

    model.add_layer(DenseLayer(X.shape[-1], 32))
    model.add_layer(RecurrentLayer(32, 256, X.shape[-1], out_act_f=ReLU()))

    model.add_layer(DenseLayer(len(char_to_int), len(char_to_int), name='Dense layer 1'))

    model.set_loss(CrossEntropy())

    model.set_optimizer(RMSProp())
    batch_size = min(50, X.shape[0])
    model.fit(Dataset(X, y), print_every=1, batch_size=batch_size, max_epochs=50, metrics=[Accuracy()])
    # model.load_params("dnn_model.pickle")
    xp.random.shuffle(X)
    """
    After training, we let the network generate text. We start by giving it a prefix so it can update
    its recurrent hidden state, then generate new text one character at a time.
    At each step we feed the last character and get a probability distribution for the next one.
    Instead of taking argmax, we sample the next character according to these probabilities.
    To support this, the sequence length is set to 1 and we run forward propagation T times,
    preserving the recurrent hidden state (_keep_h_state_for_next_sequence=True).
    """
    for l in model._layers:
        if isinstance(l, RecurrentLayer):
            l.reset_state = False
    network_input = xp.zeros((batch_size, 1, len(char_to_int)))
    # We will generate batch_size strings, and the line below creates a list of length batch_size initialized with empty strings.
    generated_text = [''] * batch_size
    print()

    warm_up = 3
    # Number of characters we feed into the network to initialize the recurrent state.
    # In this example, the first 4 characters of each generated string come from the training set,
    # while the remainder of the string is produced by the network.

    for t in range(warm_up):
        for i in range(batch_size):
            network_input[i, 0, :] = X[i, t, :]
            c = int_to_char[int(xp.argmax(network_input[i, 0, :]))]
            generated_text[i] += c
        network_output = model.forward(network_input)

    text_len = 50
    for t in range(text_len):
        for i in range(batch_size):
            # xp.random.choice() selects a random element from an array using probabilities given by p.
            # Here we pass range(len(char_to_int)), i.e., 0, 1, 2, ..., len(char_to_int) - 1.
            # c_idx = int(xp.random.choice(range(len(char_to_int)), p=Softmax()(network_output[i, 0, :]), size=1))

            c_idx = int(xp.argmax(network_output[i, 0, :]))
            generated_text[i] += int_to_char[c_idx]  # We convert the sampled index to a character and concatenate it to the generated text.
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
    DINO DATASET
    
    Training time = 0.31482362747192383 seconds
    Parameters saved at saved_models/VanillaRNN.pickle
    
    osaursaus............................................
    -----------------------
    sassus...............................................
    -----------------------
    luraueatusau.........................................
    -----------------------
    opelosaus............................................
    -----------------------
    lengussosaus.........................................
    -----------------------
    agiuresusaur.........................................
    -----------------------
    spinroaususaur.......................................
    -----------------------
    ediosraur............................................
    -----------------------
    lengussosaus.........................................
    -----------------------
    saurur...............................................
    -----------------------
    droreaosauasas.......................................
    -----------------------
    eyeausasaur..........................................
    -----------------------
    suur.................................................
    -----------------------
    ausaueurarusaur......................................
    -----------------------
    saurur...............................................
    -----------------------
    palrlousuusaus.......................................
    -----------------------
    
    
    ------------------------------------------------------
    
    USERNAMES DATASET
    
    Training time = 1.6019892692565918 seconds
    Parameters saved at saved_models/VanillaRNN.pickle
    
    milinaocte...........................................
    -----------------------
    laracotiso...........................................
    -----------------------
    igonaocuest..........................................
    -----------------------
    jovanarilon..........................................
    -----------------------
    sofiacotinner........................................
    -----------------------
    lazaraodes...........................................
    -----------------------
    maraoconnect.........................................
    -----------------------
    elenaaotes...........................................
    -----------------------
    kriarinis............................................
    -----------------------
    andrelastooies.......................................
    -----------------------
    nikelareues..........................................
    -----------------------
    katarinasloodar......................................
    -----------------------
    miainaocter..........................................
    -----------------------
    stelanrrcner.........................................
    -----------------------
    igonaocuest..........................................
    -----------------------
    saraantite...........................................
    -----------------------
    vanjamotiso..........................................
    -----------------------
    miainaocter..........................................
    -----------------------
    nenearatriner........................................
    -----------------------
    stelanrrcner.........................................
    -----------------------
    katarinasloodar......................................
    -----------------------
    milinaocte...........................................
    -----------------------
    jovanarilon..........................................
    -----------------------
    boriscriads..........................................
    -----------------------
    milinaocte...........................................
    -----------------------
    stelanrrcner.........................................
    -----------------------
    mataranater..........................................
    -----------------------
    nikelareues..........................................
    -----------------------
    tijanaoriss..........................................
    -----------------------
    lenareads............................................
    -----------------------
    dusiacantere.........................................
    -----------------------
    andrelastooies.......................................
    -----------------------
    .....................................................
    -----------------------
    anajejantarios.......................................
    -----------------------
    tamaraoiter..........................................
    -----------------------
    tinanaorisc..........................................
    -----------------------
    petaranatere.........................................
    -----------------------
    lenareads............................................
    -----------------------
    jovanarilon..........................................
    -----------------------
    andrelastooies.......................................
    -----------------------
    anajejantarios.......................................
    -----------------------
    saraantite...........................................
    -----------------------
    dariacones...........................................
    -----------------------
    tijanaoriss..........................................
    -----------------------
    miainaocter..........................................
    -----------------------
    teoanaalter..........................................
    -----------------------
    urosailion...........................................
    -----------------------
    dusiacantere.........................................
    -----------------------
    lukarajelo...........................................
    -----------------------
    vanjamotiso..........................................
    -----------------------
    """