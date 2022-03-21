from django.views import View
from django.http import HttpResponse, request
from django.shortcuts import render, get_object_or_404
from base64 import b64decode
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
        }

        return render(request, "base.html", context)

    def post(self, request, *args, **kwargs):
        print("post")
        form = UploadImageForm(request.POST, request.FILES)
        app = apps.get_app_config("neuralngin")

        if form.is_valid():
            imagefile: InMemoryUploadedFile = request.FILES["image"]
            output = app.predict(imagefile.file.read(), form.cleaned_data["model"])
            image = "data:image/jpeg;base64," + str(output)[2:-2]

            return render(request, "output.html", {"image": image})

        print(form.errors)


class SecondView(View):
    def get(self, request):
        print("get")
        return render(request, "output.html")
