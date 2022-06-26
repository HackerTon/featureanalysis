# Analysis of My Feature Map

## Prerequisite

**IMPORTANT !!**

Before you run the training code below, make sure you have install Tensorflow V2 and above and Segmentation_models library. To install, run the command below

```bash
pip install segmentation-models
```

## Experimental results

Uses dice loss and with pixelwise crossentropy

| Dataset | Batch Size | Learning Rate | Training Dataset Size | Testing Dataset Size | Image Size (W x H) |
| ------- | ---------- | ------------- | --------------------- | -------------------- | ------------------ |
| UAVid   | 8          | 1e-5          | 18499                 | 4624                 | 256 x 256          |
| Seagull | 8          | 1e-5          | 6400                  | 2240                 | 448 x 448          |

### UAvid Dataset

| MODEL                      | BACKBONE       | Train IoU | Test IoU |
| -------------------------- | -------------- | --------- | -------- |
| FPN                        | efficientNetB0 | 0.604     | 0.476    |
| UNET                       | efficientNetB0 | 0.516     | 0.405    |
| FCN                        | efficientNetB0 | 0.258     | 0.192    |
| UNET + FPN (PRODUCT)       | efficientNetB0 | 0.639     | 0.566    |
| UNET + FPN (SUM)           | efficientNetB0 | 0.589     | 0.523    |
| UNET + FPN (CONCATENATION) | efficientNetB0 | 0.646     | 0.574    |
| FCN + FPN (CONCATENATION)  | efficientNetB0 | 0.394     | 0.446    |

### Seagull Dataset

| MODEL                      | BACKBONE       | Train IoU | Test IoU |
| -------------------------- | -------------- | --------- | -------- |
| FPN                        | efficientNetB0 | 0.841     | 0.834    |
| UNET                       | efficientNetB0 | 0.825     | 0.824    |
| FCN                        | efficientNetB0 | 0.494     | 0.498    |
| UNET + FPN (PRODUCT)       | efficientNetB0 | TBA       | TBA      |
| UNET + FPN (SUM)           | efficientNetB0 | TBA       | TBA      |
| UNET + FPN (CONCATENATION) | efficientNetB0 | TBA       | TBA      |
| FCN + FPN (CONCATENATION)  | efficientNetB0 | TBA       | TBA      |
