from django.views import View
from django.http import HttpResponse
from django.shortcuts import render


class Mainview(View):
    def get(self, request, *args, **kwargs):
        return render(request, "base.html")
