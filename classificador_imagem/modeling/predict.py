import numpy as np
from numpy.typing import NDArray
from tensorflow import keras


def predict_images(
    model: keras.Model, images: NDArray
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Return class probabilities and the most likely class for each image."""
    probabilities = model.predict(images, verbose=0)
    labels = np.argmax(probabilities, axis=1)
    return probabilities, labels
