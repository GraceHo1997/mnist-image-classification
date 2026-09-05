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

## Local Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate