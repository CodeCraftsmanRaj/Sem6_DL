# model.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import config


def build_model():
    model = Sequential([
        Embedding(input_dim=config.VOCAB_SIZE,
                  output_dim=config.EMBEDDING_DIM,
                  input_length=config.MAX_LENGTH),

        SimpleRNN(config.RNN_UNITS),

        Dense(1, activation='sigmoid')
    ])

    return model