from django import forms
from django.conf import settings
import os

MODEL_CHOICE = (("1", "FCN-8s"), ("2", "UNet"), ("3", "FPN"), ("4", "Ensemble"))


class UploadImageForm(forms.Form):
    model = forms.ChoiceField(choices=MODEL_CHOICE)
    image = forms.ImageField()

    # configure the field with css class
    model.widget.attrs.update({"class": "model"})
    image.widget.attrs.update({"onchange": "readURL(this)", "class": "image"})
