from io import BytesIO

import numpy as np
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, TestCase
from django.urls import reverse

from .services import ImageValidationError, process_csv
from unittest.mock import patch


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
            
            
class ClassifierViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="dan",
            password="Optimization1234",
        )
        self.home_url = reverse("home")

    def test_logged_out_user_is_redirected_to_login(self):
        response = self.client.get(self.home_url)

        self.assertRedirects(
            response,
            f"/accounts/login/?next={self.home_url}",
        )

    def test_logged_in_user_can_open_upload_page(self):
        self.client.login(
            username="dan",
            password="Optimization1234",
        )

        response = self.client.get(self.home_url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "MNIST Digit Classifier")

    @patch("classifier.views.predict_image")
    def test_valid_csv_can_be_uploaded(self, mock_predict_image):
        mock_predict_image.return_value = {
            "predicted_digit": 8,
            "confidence": 100.0,
            "probabilities": [
                {"digit": digit, "probability": 0.0}
                for digit in range(10)
            ],
        }

        self.client.login(
            username="dan",
            password="Optimization1234",
        )

        csv_buffer = create_csv_file(np.zeros((28, 28)))
        uploaded_file = SimpleUploadedFile(
            "digit.csv",
            csv_buffer.getvalue(),
            content_type="text/csv",
        )

        response = self.client.post(
            self.home_url,
            {"csv_file": uploaded_file},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Prediction Result")
        self.assertContains(response, "Predicted digit: 8")
        mock_predict_image.assert_called_once()

    def test_non_csv_file_is_rejected(self):
        self.client.login(
            username="dan",
            password="Optimization1234",
        )

        uploaded_file = SimpleUploadedFile(
            "digit.txt",
            b"not a csv file",
            content_type="text/plain",
        )

        response = self.client.post(
            self.home_url,
            {"csv_file": uploaded_file},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "File extension")

    def test_start_over_returns_clean_form(self):
        self.client.login(
            username="dan",
            password="Optimization1234",
        )

        response = self.client.get(self.home_url)

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Prediction Result")