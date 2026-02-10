import tensorflow as tf
from config.config import *

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

model = tf.keras.models.load_model("models/transfer_model.keras")
loss, accuracy = model.evaluate(test_ds)

print("Test Accuracy:", accuracy)
