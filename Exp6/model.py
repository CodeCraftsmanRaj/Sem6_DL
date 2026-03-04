# model.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
import config


def build_model():
    model = Sequential([
        Embedding(input_dim=config.VOCAB_SIZE,
                  output_dim=config.EMBEDDING_DIM),

        LSTM(config.RNN_UNITS,
             dropout=0.3,
             recurrent_dropout=0.3),

        Dropout(0.5),

        Dense(1, activation='sigmoid',
              kernel_regularizer=l2(0.001))
    ])

    return model