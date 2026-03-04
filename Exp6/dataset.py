# dataset.py

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import config


def load_data():
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=config.VOCAB_SIZE)

    X_train = pad_sequences(X_train, maxlen=config.MAX_LENGTH)
    X_test = pad_sequences(X_test, maxlen=config.MAX_LENGTH)

    return X_train, X_test, y_train, y_test