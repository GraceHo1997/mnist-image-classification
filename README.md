# MNIST Image Classification

A Django web application that allows authenticated users to upload a 28 × 28 CSV image and classify the handwritten digit using a TensorFlow CNN model.

## Current Features

- Login and logout
- Protected classifier page
- CSV file upload
- 28 × 28 dimension validation
- Numeric and pixel-range validation
- Automatic 0–1 and 0–255 scale detection
- Image preview
- Start Over functionality
- Automated tests

## Requirements

- Python 3.11
- TensorFlow 2.20.0
- Keras 3.13.2

## Local Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate

## Model

The application uses `mnist_classifier_group5.keras`, trained with TensorFlow 2.20.0 and Keras 3.13.2.

- Input shape: `(None, 28, 28, 1)`
- Input type: `float32`
- Pixel scale: `0–1`
- Output: 10 probabilities for digits 0–9

The application automatically normalizes CSV values from 0–255 when needed.