from django.views import View
from django.http import HttpResponse, request
from django.shortcuts import render, get_object_or_404
from base64 import b64decode
from django.apps import apps
from .forms import UploadImageForm
from django.core.files.uploadedfile import InMemoryUploadedFile


class Mainview(View):
    # app = apps.get_app_config("neuralngin")

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

        if form.is_valid():
            imagefile: InMemoryUploadedFile = request.FILES["image"]
            print(imagefile.file.read())
