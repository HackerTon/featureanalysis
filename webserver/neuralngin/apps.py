# import datetime
# import functools as ft
# import glob
# import os

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sb
# import segmentation_models as sm
# import tensorflow as tf
# from django.apps import AppConfig
# from django.conf import settings

# # disable GPU computation
# tf.config.set_visible_devices([], "GPU")
# sm.set_framework("tf.keras")
# sm.framework()

# tf.random.set_seed(1024)
# SEED = 100


# class NeuralnginConfig(AppConfig):
#     name = "neuralngin"

#     def ready(self):
#         self.fcn = tf.keras.models.load_model(
#             os.path.join(settings.BASE_DIR, "savedmodel/seagull_fpn")
#         )

#     def predict(self, model):
#         return "die"
