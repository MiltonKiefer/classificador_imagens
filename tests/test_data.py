import numpy as np

from classificador_imagem.features import preprocess_split


def test_preprocess_split_scales_images_and_flattens_labels():
    images = np.array(
        [
            [[[0, 127, 255]]],
            [[[255, 0, 127]]],
        ],
        dtype=np.uint8,
    )
    labels = np.array([[2], [7]], dtype=np.uint8)

    processed_images, processed_labels = preprocess_split(images, labels)

    assert processed_images.dtype == np.float32
    np.testing.assert_allclose(
        processed_images,
        np.array(
            [
                [[[0.0, 127 / 255, 1.0]]],
                [[[1.0, 0.0, 127 / 255]]],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(processed_labels, np.array([2, 7], dtype=np.uint8))


def test_preprocess_split_preserves_image_shape():
    images = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    labels = np.zeros((4, 1), dtype=np.uint8)

    processed_images, processed_labels = preprocess_split(images, labels)

    assert processed_images.shape == (4, 32, 32, 3)
    assert processed_labels.shape == (4,)
