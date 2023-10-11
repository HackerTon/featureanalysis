from django.views import View
from django.http import HttpResponse, request
from django.shortcuts import render, get_object_or_404
from base64 import b64decode, b64encode
from django.apps import apps
from .forms import UploadImageForm
from django.core.files.uploadedfile import InMemoryUploadedFile


class Mainview(View):
    def get(self, request, *args, **kwargs):
        print("get")

        form = UploadImageForm()
        context = {
            "model": form["model"],
            "image": form["image"],
            "infclass": form["infclass"],
        }

        return render(request, "base.html", context)

    def post(self, request, *args, **kwargs):
        print("post")
        form = UploadImageForm(request.POST, request.FILES)
        app = apps.get_app_config("neuralngin")

        if form.is_valid():
            imagefile: InMemoryUploadedFile = request.FILES["image"]
            binary_img = imagefile.file.read()

            output = app.predict(
                binary_img,
                form.cleaned_data["model"],
                form.cleaned_data["infclass"],
            )
            imageoutput = r"data:image/jpeg;base64," + str(output)[2:-1]
            imageinput = r"data:image/jpeg;base64," + str(b64encode(binary_img))[2:-1]

            return render(
                request,
                "output.html",
                {
                    "outputimage": imageoutput,
                    "inputimage": imageinput,
                    "type": dict(form.fields["model"].choices)[
                        form.cleaned_data["model"]
                    ],
                },
                # {"inputimage": imageinput},
            )

        print(form.errors)


class SecondView(View):
    def get(self, request):
        print("get")
        return render(request, "output.html")
