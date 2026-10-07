from tensorflow import keras

from classificador_imagem.config import BATCH_SIZE, EPOCHS, VALIDATION_SPLIT


def build_callbacks() -> list[keras.callbacks.Callback]:
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
        ),
    ]


def train_model(model: keras.Model, images, labels) -> keras.callbacks.History:
    """Train the model with the notebook's settings and validation callbacks."""
    return model.fit(
        images,
        labels,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=build_callbacks(),
    )
