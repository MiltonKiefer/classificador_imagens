from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D

from classificador_imagem.config import IMAGE_SHAPE


def build_model() -> keras.Model:
    """Build and compile the CIFAR-10 convolutional classifier."""
    model = keras.Sequential(
        [
            Input(shape=IMAGE_SHAPE),
            Conv2D(filters=16, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
