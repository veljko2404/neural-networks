import pandas as pd
from loss_functions.mse import MSE
from models.feedforward_nn import Model
from layers.dense_layer import DenseLayer
from layers.recurrent_layers.gru import GRU
from layers.recurrent_layers.lstm import LSTM
from optimizers.adam import Adam
from utils.dataset import Dataset
from backend.backend import xp

def generate_price_sequences(prices, seq_len=60):
    X, y = [], []
    for i in range(len(prices) - seq_len):
        X.append(prices[i:i + seq_len])
        y.append(prices[i + seq_len])
    return xp.array(X), xp.array(y)

def btc_price_lstm():
    model = Model(name="btc_prices_lstm")

    df = pd.read_csv("data/btc_prices.csv", sep=";")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    prices = df["close"].values.astype(xp.float32)

    log_returns = xp.log(prices[1:] / prices[:-1])

    mean_r = log_returns.mean()
    std_r = log_returns.std() + 1e-8

    log_returns = (log_returns - mean_r) / std_r

    SEQ_LEN = 60
    X, y = generate_price_sequences(log_returns, SEQ_LEN)

    X = X[..., xp.newaxis]
    y = y[..., xp.newaxis]
    y = xp.repeat(y[:, xp.newaxis, :], SEQ_LEN, axis=1)

    model.add_layer(DenseLayer(1, 32))
    # model.add_layer(GRU(32, 128))
    model.add_layer(LSTM(32, 128))
    model.add_layer(DenseLayer(128, 1))

    model.set_loss(MSE())
    model.set_optimizer(Adam())

    batch_size = min(32, X.shape[0])

    model.fit(Dataset(X, y), batch_size=batch_size, max_epochs=24, print_every=4)

    for l in model._layers:
        if isinstance(l, LSTM) or isinstance(l, GRU):
            l.reset_state = False

    last_seq = X[-1:,:,:]

    future_steps = 7
    predictions = []

    for _ in range(future_steps):
        pred = model.forward(last_seq)
        pred_val = pred[0, -1, 0]

        predictions.append(pred_val)

        last_seq = xp.roll(last_seq, -1, axis=1)
        last_seq[0, -1, 0] = pred_val

    last_price = prices[-1]
    predicted_prices = []

    for r_hat in predictions:
        r = r_hat * std_r + mean_r
        last_price = last_price * xp.exp(r)
        predicted_prices.append(last_price)

    print("Predicted BTC prices:")
    for i, p in enumerate(predicted_prices, 1):
        if i == 1:
            print(f"Day +{i}: {p:.2f} $, {p-last_price:.2f} $")
        else:
            print(f"Day +{i}: {p:.2f} $, {p-predicted_prices[i-2]:.2f} $")

    """
    Training:	Epoch: 20, loss: 0.5048355937744552.
    --------------------------------------------------
    
    Training:	Epoch: 24, loss: 0.4974957468725078.
    --------------------------------------------------
    
    Training time = 14.242902755737305 seconds
    Parameters saved at saved_models/params_btc_prices_lstm.pickle
    Model saved at saved_models/model_btc_prices_lstm.pickle
    
    Predicted BTC prices:
    Day +1: 88060.27 $, 1119.91 $
    Day +2: 87989.64 $, -70.64 $
    Day +3: 87863.93 $, -125.70 $
    Day +4: 87679.08 $, -184.86 $
    Day +5: 87449.89 $, -229.19 $
    Day +6: 87197.31 $, -252.58 $
    Day +7: 86940.37 $, -256.94 $
    
    """
