from tensorflow import keras


def evaluate_model(model: keras.Model, images, labels) -> tuple[float, float]:
    """Return test loss and accuracy."""
    loss, accuracy = model.evaluate(images, labels, verbose=1)
    return float(loss), float(accuracy)
