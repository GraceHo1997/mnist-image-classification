import numpy as np
import pandas as pd
import base64
from io import BytesIO
from PIL import Image


class ImageValidationError(Exception):
    """Raised when an uploaded CSV is not a valid MNIST image."""


def process_csv(uploaded_file):
    try:
        dataframe = pd.read_csv(uploaded_file, header=None)
    except Exception as error:
        raise ImageValidationError("The uploaded file could not be read as a CSV.") from error

    if dataframe.shape != (28, 28):
        raise ImageValidationError(
            "The CSV file must contain exactly 28 rows and 28 columns."
        )

    try:
        image = dataframe.apply(
            pd.to_numeric,
            errors="raise",
        ).to_numpy(dtype=np.float32)
    except (ValueError, TypeError) as error:
        raise ImageValidationError(
            "Every entry in the CSV file must be numeric."
        ) from error

    if not np.isfinite(image).all():
        raise ImageValidationError(
            "The CSV file cannot contain blank or infinite values."
        )

    minimum = float(image.min())
    maximum = float(image.max())

    if minimum < 0 or maximum > 255:
        raise ImageValidationError(
            "Pixel values must be between 0 and 255."
        )

    # Files with values above 1 use the 0–255 scale.
    if maximum > 1:
        image = image / 255.0

    image_tensor = image.reshape(1, 28, 28, 1)

    return image, image_tensor

def image_to_data_url(image):
    pixel_image = (image * 255).astype(np.uint8)

    preview = Image.fromarray(pixel_image, mode="L")
    preview = preview.resize((280, 280), Image.Resampling.NEAREST)

    buffer = BytesIO()
    preview.save(buffer, format="PNG")

    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/png;base64,{encoded_image}"