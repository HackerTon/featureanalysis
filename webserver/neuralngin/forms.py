from django import forms
from django.conf import settings
import os


class UploadImageForm(forms.Form):
    MODEL_CHOICE = (
        ("fcn", "FCN-8s"),
        ("unet", "UNet"),
        ("fpn", "FPN"),
        ("ensem", "Ensemble"),
    )
    model = forms.ChoiceField(choices=MODEL_CHOICE)
    infclass = forms.IntegerField(min_value=0, max_value=7)
    image = forms.ImageField()

    # configure the field with css class
    model.widget.attrs.update({"class": "model"})
    model.widget.attrs.update({"class": "model"})
    image.widget.attrs.update({"onchange": "readURL(this)", "class": "image"})
