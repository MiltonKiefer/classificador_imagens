import numpy as np
from numpy.typing import NDArray


def preprocess_split(images: NDArray, labels: NDArray) -> tuple[NDArray[np.float32], NDArray]:
    """Scale image pixels to [0, 1] and flatten CIFAR-10 labels."""
    normalized_images = images.astype("float32") / 255.0
    flattened_labels = labels.flatten()
    return normalized_images, flattened_labels
