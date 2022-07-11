from django import forms
from django.conf import settings


class UploadImageForm(forms.Form):
    MODEL_CHOICE = (
        ("fcnseagull", "FCN-8s seagull"),
        ("unetseagull", "U-NET seagull"),
        ("fpnseagull", "FPN seagull"),
        ("ufpnprodseagull", "U-NET + FPN (PROD) seagull"),
        ("ufpnsumseagull", "U-NET + FPN (SUM) seagull"),
        ("ufpnconcatseagull", "U-NET + FPN (CONCAT) seagull"),
        ("ffpnprodseagull", "FCN-8s + FPN (PROD) seagull"),
        ("fcnuavid", "FCN-8s uavid"),
        ("unetuavid", "U-NET uavid"),
        ("fpnuavid", "FPN uavid"),
        ("ufpnproduavid", "U-NET + FPN (PROD) uavid"),
        ("ufpnsumuavid", "U-NET + FPN (SUM) uavid"),
        ("ufpnconcatuavid", "U-NET + FPN (CONCAT) uavid"),
        ("ffpnproduavid", "FCN8s + FPN (PROD) uavid"),
    )

    model = forms.ChoiceField(choices=MODEL_CHOICE)
    infclass = forms.IntegerField(min_value=0, max_value=7)
    image = forms.ImageField()

    # configure the field with css class
    model.widget.attrs.update({"class": "model"})
    model.widget.attrs.update({"class": "model"})
    image.widget.attrs.update({"onchange": "readURL(this)", "class": "image"})
