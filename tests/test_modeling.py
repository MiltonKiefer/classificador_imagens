import pytest


def test_model_has_cifar10_input_and_ten_class_output():
    pytest.importorskip("tensorflow")

    from classificador_imagem.modeling.model import build_model

    model = build_model()

    assert model.input_shape == (None, 32, 32, 3)
    assert model.output_shape == (None, 10)
    assert model.loss == "sparse_categorical_crossentropy"
    assert model.optimizer.name == "adam"


def test_training_callbacks_are_configured_for_validation_loss():
    pytest.importorskip("tensorflow")

    from classificador_imagem.modeling.train import build_callbacks

    callbacks = build_callbacks()

    assert [callback.monitor for callback in callbacks] == ["val_loss", "val_loss"]
    assert callbacks[0].patience == 5
    assert callbacks[0].restore_best_weights is True
    assert callbacks[1].patience == 2
    assert callbacks[1].factor == 0.5
