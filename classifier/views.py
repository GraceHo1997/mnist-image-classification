from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from .forms import CSVUploadForm
from .services import (
    ImageValidationError,
    image_to_data_url,
    process_csv,
)
from .model_service import predict_image

@login_required
def home(request):
    result = None

    if request.method == "POST":
        form = CSVUploadForm(request.POST, request.FILES)

        if form.is_valid():
            try:
                image, image_tensor = process_csv(
                    form.cleaned_data["csv_file"]
                )

                prediction = predict_image(image_tensor)

                result = {
                    **prediction,
                    "image_url": image_to_data_url(image),
                }

            except ImageValidationError as error:
                form.add_error("csv_file", str(error))
    else:
        form = CSVUploadForm()

    return render(
        request,
        "classifier/upload.html",
        {
            "form": form,
            "result": result,
        },
    )


@login_required
def process_article(request):
    return render(request, "classifier/article.html")
