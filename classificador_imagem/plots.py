import matplotlib.pyplot as plt
from numpy.typing import NDArray

from classificador_imagem.config import CLASS_NAMES, PLOT_SAMPLE_COUNT


def plot_samples(images: NDArray, labels: NDArray) -> None:
    figure = plt.figure(figsize=(10, 10))
    for index in range(PLOT_SAMPLE_COUNT):
        plt.subplot(3, 3, index + 1)
        plt.imshow(images[index])
        plt.title(CLASS_NAMES[int(labels[index])])
        plt.axis("off")
    figure.suptitle("Exemplos da base CIFAR-10")
    plt.show()


def plot_training_history(history) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["accuracy"], label="Treino")
    axes[0].plot(history.history["val_accuracy"], label="Validação")
    axes[0].set(title="Acurácia por época", xlabel="Épocas", ylabel="Acurácia")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Treino")
    axes[1].plot(history.history["val_loss"], label="Validação")
    axes[1].set(title="Loss por época", xlabel="Épocas", ylabel="Loss")
    axes[1].legend()
    figure.tight_layout()
    plt.show()


def plot_predictions(
    images: NDArray,
    true_labels: NDArray,
    predicted_labels: NDArray,
    probabilities: NDArray,
) -> None:
    figure = plt.figure(figsize=(10, 10))
    for index in range(PLOT_SAMPLE_COUNT):
        plt.subplot(3, 3, index + 1)
        plt.imshow(images[index])
        real = CLASS_NAMES[int(true_labels[index])]
        predicted = CLASS_NAMES[int(predicted_labels[index])]
        confidence = probabilities[index].max()
        plt.title(f"Real: {real}\nPredição: {predicted}\nProbabilidade: {confidence:.2f}")
        plt.axis("off")
    figure.suptitle("Previsões em imagens do conjunto de teste")
    figure.tight_layout()
    plt.show()
