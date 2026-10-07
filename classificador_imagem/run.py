from classificador_imagem.dataset import load_cifar10
from classificador_imagem.features import preprocess_split
from classificador_imagem.modeling.evaluate import evaluate_model
from classificador_imagem.modeling.model import build_model
from classificador_imagem.modeling.predict import predict_images
from classificador_imagem.modeling.train import train_model
from classificador_imagem.plots import plot_predictions, plot_samples, plot_training_history


def main() -> None:
    (train_images, train_labels), (test_images, test_labels) = load_cifar10()
    train_images, train_labels = preprocess_split(train_images, train_labels)
    test_images, test_labels = preprocess_split(test_images, test_labels)

    plot_samples(train_images, train_labels)

    model = build_model()
    model.summary()
    history = train_model(model, train_images, train_labels)
    plot_training_history(history)

    test_loss, test_accuracy = evaluate_model(model, test_images, test_labels)
    print(f"Acurácia no conjunto de teste: {test_accuracy:.4f}")
    print(f"Loss no teste: {test_loss:.4f}")

    probabilities, predicted_labels = predict_images(model, test_images[:9])
    plot_predictions(
        test_images[:9],
        test_labels[:9],
        predicted_labels,
        probabilities,
    )


if __name__ == "__main__":
    main()
