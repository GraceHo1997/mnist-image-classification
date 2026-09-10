from functools import lru_cache

import numpy as np
from django.conf import settings
from tensorflow import keras


MODEL_PATH = (
    settings.BASE_DIR
    / "models"
    / "mnist_classifier_group5.keras"
)


@lru_cache(maxsize=1)
def get_model():
    """Load the model once and reuse it for later predictions."""
    return keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )


def predict_image(image_tensor):
    model = get_model()

    probabilities = model.predict(
        image_tensor,
        verbose=0,
    )[0]

    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit] * 100)

    probability_list = [
        {
            "digit": digit,
            "probability": float(probability * 100),
        }
        for digit, probability in enumerate(probabilities)
    ]

    return {
        "predicted_digit": predicted_digit,
        "confidence": confidence,
        "probabilities": probability_list,
    }