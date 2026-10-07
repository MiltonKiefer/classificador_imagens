def load_cifar10():
    """Load the CIFAR-10 train and test splits using Keras."""
    from tensorflow import keras

    return keras.datasets.cifar10.load_data()
