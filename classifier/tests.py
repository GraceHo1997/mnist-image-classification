from io import BytesIO

import numpy as np
from django.test import SimpleTestCase

from .services import ImageValidationError, process_csv


def create_csv_file(array):
    content = "\n".join(
        ",".join(str(value) for value in row)
        for row in array
    )
    return BytesIO(content.encode("utf-8"))


class ProcessCSVTests(SimpleTestCase):
    def test_accepts_zero_to_one_scale(self):
        uploaded_file = create_csv_file(np.full((28, 28), 0.5))

        image, tensor = process_csv(uploaded_file)

        self.assertEqual(image.shape, (28, 28))
        self.assertEqual(tensor.shape, (1, 28, 28, 1))
        self.assertAlmostEqual(float(image.max()), 0.5)

    def test_normalizes_zero_to_255_scale(self):
        uploaded_file = create_csv_file(np.full((28, 28), 255))

        image, tensor = process_csv(uploaded_file)

        self.assertAlmostEqual(float(image.max()), 1.0)
        self.assertEqual(tensor.shape, (1, 28, 28, 1))

    def test_rejects_wrong_dimensions(self):
        uploaded_file = create_csv_file(np.zeros((27, 28)))

        with self.assertRaisesMessage(
            ImageValidationError,
            "exactly 28 rows and 28 columns",
        ):
            process_csv(uploaded_file)

    def test_rejects_text_values(self):
        array = np.zeros((28, 28), dtype=object)
        array[0, 0] = "invalid"
        uploaded_file = create_csv_file(array)

        with self.assertRaisesMessage(
            ImageValidationError,
            "must be numeric",
        ):
            process_csv(uploaded_file)

    def test_rejects_negative_values(self):
        array = np.zeros((28, 28))
        array[0, 0] = -1
        uploaded_file = create_csv_file(array)

        with self.assertRaisesMessage(
            ImageValidationError,
            "between 0 and 255",
        ):
            process_csv(uploaded_file)

    def test_rejects_values_above_255(self):
        array = np.zeros((28, 28))
        array[0, 0] = 256
        uploaded_file = create_csv_file(array)

        with self.assertRaisesMessage(
            ImageValidationError,
            "between 0 and 255",
        ):
            process_csv(uploaded_file)