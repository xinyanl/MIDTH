from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input, Model
from matplotlib import pyplot as plt 
from helper_functions import *

def fit_lstm(
    X_train, y_train, X_test, y_test, batch_size, nb_epoch, neurons, y_size=1, verbose=0, n_cycle=15
):
    """
    Takes training data and test data and a few model hyperparameters,
    trains a LSTM model and plots the results
    returns the trained model and a parity plot 

    Parameters:
        X_train: numpy.array, the feature values in training data
        y_train: numpy.array, the target values in training data
        X_test: numpy.array, the feature values in test data
        y_test: numpy.array, the target values in test data
        nb_epoch: int, number of epochs to train
        batch_size: int, number of batch size
        neurons: int, number of neurons to train
        n_cycle: int, default=15, use the data from the 6th to the 20th cycle 

    Returns:
        model: tf.Model, the trained model
        VIANN: helper_functions.VarImpVIANN,
               variable importance analysis callback that tracks and 
               computes feature importance scores during model training 
    """
    seq_input = Input(batch_shape=(batch_size, X.shape[1], X.shape[2]))
    l1 = LSTM(
        neurons, 
        stateful=False,
        dropout=0.2, 
        recurrent_dropout=0.2,
    )(seq_input)
    l2 = Dense(neurons, activation="relu")(l1)
    l = Dense(y_size)(l2)
    VIANN = VarImpVIANN(verbose=1)
    model = Model(seq_input, l)
    model.compile(
        loss="mean_absolute_error", 
        optimizer=Adam(learning_rate=1e-3)
    )
    model.fit(
        X_train, y_train, epochs=nb_epoch, batch_size=batch_size, 
        verbose=verbose, shuffle=True, callbacks=[VIANN],
        validation_split=0.2
    )
    train_pred, test_pred = (model.predict(X_train), model.predict(X_test))
    train_err, test_err = (
        np.mean(np.abs((n_cycle/train_pred.flatten() - n_cycle/y_train.flatten()) / (n_cycle/y_train.flatten()))),
        np.mean(np.abs((n_cycle/test_pred.flatten() - n_cycle/y_test.flatten()) / (n_cycle/y_test.flatten()))),
    )
    fig, ax = plt.subplots()
    lim = (0., 180)
    ax.plot(
        n_cycle/y_train.flatten(), n_cycle/train_pred.flatten(), marker="o", linewidth=0,
        label="Train: %.3f" %train_err
    )
    ax.plot(
        n_cycle/y_test.flatten(), n_cycle/test_pred.flatten(), marker="o", linewidth=0,
        label="Test: %.3f" %test_err
    )
    ax.plot(lim, lim, color="k")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.legend()
    ax.set_xlabel("Actual EP")
    ax.set_ylabel("Predicted EP")    
    return model, VIANN
