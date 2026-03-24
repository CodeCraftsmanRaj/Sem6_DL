import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "experiment_outputs"
SAMPLES_DIR = OUTPUT_DIR / "generated_samples"
OUTPUT_DIR.mkdir(exist_ok=True)
SAMPLES_DIR.mkdir(exist_ok=True)

SEED = 42
LATENT_DIM = 100
BATCH_SIZE = 256
EPOCHS = 10
TRAINING_SAMPLES = 10000
SAVE_IMAGE_EVERY = 2

keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


class MNISTGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.seed_generator = keras.random.SeedGenerator(SEED)
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        random_latent_vectors = keras.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )
        generated_images = self.generator(random_latent_vectors, training=True)

        combined_images = tf.concat([generated_images, real_images], axis=0)
        labels = tf.concat(
            [
                tf.zeros((batch_size, 1)),
                tf.ones((batch_size, 1)) * 0.9,
            ],
            axis=0,
        )
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images, training=True)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        misleading_labels = tf.ones((batch_size, 1))
        random_latent_vectors = keras.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_latent_vectors, training=True)
            predictions = self.discriminator(fake_images, training=False)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        return {"d_loss": self.d_loss_tracker.result(), "g_loss": self.g_loss_tracker.result()}


class SampleImageCallback(keras.callbacks.Callback):
    def __init__(self, latent_dim, save_every, output_dir):
        super().__init__()
        self.latent_dim = latent_dim
        self.save_every = save_every
        self.output_dir = output_dir
        self.fixed_latent_vectors = tf.random.normal((16, latent_dim), seed=SEED)

    def on_epoch_end(self, epoch, logs=None):
        epoch_number = epoch + 1
        if epoch_number == 1 or epoch_number % self.save_every == 0 or epoch_number == EPOCHS:
            generated_images = self.model.generator(self.fixed_latent_vectors, training=False)
            generated_images = (generated_images * 127.5) + 127.5
            generated_images = tf.clip_by_value(generated_images, 0, 255).numpy().astype("uint8")
            save_image_grid(
                generated_images,
                self.output_dir / f"generated_epoch_{epoch_number:03d}.png",
                title=f"Generated MNIST Digits - Epoch {epoch_number}",
            )


class HistorySaver(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.history_rows = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.history_rows.append(
            {
                "epoch": epoch + 1,
                "d_loss": float(logs.get("d_loss", 0.0)),
                "g_loss": float(logs.get("g_loss", 0.0)),
            }
        )


def save_image_grid(images, output_path, title):
    plt.figure(figsize=(6, 6))
    for index in range(16):
        plt.subplot(4, 4, index + 1)
        plt.imshow(images[index].squeeze(), cmap="gray")
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def build_generator(latent_dim):
    model = keras.Sequential(
        [
            layers.Input(shape=(latent_dim,)),
            layers.Dense(7 * 7 * 128, use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Reshape((7, 7, 128)),
            layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv2D(1, kernel_size=7, padding="same", activation="tanh"),
        ],
        name="generator",
    )
    return model


def build_discriminator():
    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Dropout(0.3),
            layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="discriminator",
    )
    return model


def save_model_summary(model, output_path):
    buffer = io.StringIO()
    model.summary(print_fn=lambda line: buffer.write(line + "\n"))
    output_path.write_text(buffer.getvalue(), encoding="utf-8")


def main():
    print("\nLoading MNIST dataset...\n")
    (x_train, _), (_, _) = keras.datasets.mnist.load_data()
    x_train = x_train[:TRAINING_SAMPLES].astype("float32")
    x_train = (x_train - 127.5) / 127.5
    x_train = np.expand_dims(x_train, axis=-1)

    pd.DataFrame(
        {
            "Metric": ["Training Samples", "Image Height", "Image Width", "Channels"],
            "Value": [x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3]],
        }
    ).to_csv(OUTPUT_DIR / "dataset_info.csv", index=False)

    sample_real = ((x_train[:16] * 127.5) + 127.5).astype("uint8")
    save_image_grid(sample_real, OUTPUT_DIR / "real_mnist_samples.png", "Real MNIST Samples")

    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    dataset = dataset.shuffle(buffer_size=1024, seed=SEED).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    generator = build_generator(LATENT_DIM)
    discriminator = build_discriminator()

    print("\nGENERATOR ARCHITECTURE\n")
    generator.summary()
    print("\nDISCRIMINATOR ARCHITECTURE\n")
    discriminator.summary()

    save_model_summary(generator, OUTPUT_DIR / "generator_summary.txt")
    save_model_summary(discriminator, OUTPUT_DIR / "discriminator_summary.txt")

    gan = MNISTGAN(discriminator=discriminator, generator=generator, latent_dim=LATENT_DIM)
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        loss_fn=keras.losses.BinaryCrossentropy(),
    )

    history_saver = HistorySaver()
    history = gan.fit(
        dataset,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[
            SampleImageCallback(LATENT_DIM, SAVE_IMAGE_EVERY, SAMPLES_DIR),
            history_saver,
        ],
    )

    history_df = pd.DataFrame(history_saver.history_rows)
    history_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(history_df["epoch"], history_df["d_loss"], label="Discriminator Loss")
    plt.plot(history_df["epoch"], history_df["g_loss"], label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Training Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_loss_plot.png", dpi=300)
    plt.close()

    fixed_noise = tf.random.normal((16, LATENT_DIM), seed=SEED + 7)
    final_images = generator(fixed_noise, training=False)
    final_images = ((final_images * 127.5) + 127.5).numpy().astype("uint8")
    save_image_grid(final_images, OUTPUT_DIR / "final_generated_images.png", "Final Generated MNIST Images")

    generator.save(OUTPUT_DIR / "generator_model.keras")
    discriminator.save(OUTPUT_DIR / "discriminator_model.keras")

    final_metrics = pd.DataFrame(
        {
            "Metric": [
                "Final Discriminator Loss",
                "Final Generator Loss",
                "Best Discriminator Loss",
                "Best Generator Loss",
                "Epochs Trained",
                "Latent Dimension",
                "Batch Size",
                "Training Samples",
            ],
            "Value": [
                float(history_df["d_loss"].iloc[-1]),
                float(history_df["g_loss"].iloc[-1]),
                float(history_df["d_loss"].min()),
                float(history_df["g_loss"].min()),
                int(history_df["epoch"].iloc[-1]),
                LATENT_DIM,
                BATCH_SIZE,
                TRAINING_SAMPLES,
            ],
        }
    )
    final_metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    with open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8") as file:
        file.write("Experiment 9: GAN on MNIST\n")
        file.write(f"Training samples: {TRAINING_SAMPLES}\n")
        file.write(f"Epochs: {EPOCHS}\n")
        file.write(f"Latent dimension: {LATENT_DIM}\n")
        file.write(f"Batch size: {BATCH_SIZE}\n\n")
        file.write("Final Metrics\n")
        for metric_name, metric_value in final_metrics.itertuples(index=False):
            file.write(f"{metric_name}: {metric_value}\n")

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump({row["Metric"]: row["Value"] for _, row in final_metrics.iterrows()}, file, indent=2)

    print("\nExperiment 9 completed successfully.\n")
    print(final_metrics)


if __name__ == "__main__":
    main()
