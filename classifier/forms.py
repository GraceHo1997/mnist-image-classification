from django import forms
from django.core.validators import FileExtensionValidator


class CSVUploadForm(forms.Form):
    csv_file = forms.FileField(
        label="Upload a 28 × 28 CSV file",
        validators=[FileExtensionValidator(allowed_extensions=["csv"])],
        widget=forms.ClearableFileInput(
            attrs={"accept": ".csv"}
        ),
    )