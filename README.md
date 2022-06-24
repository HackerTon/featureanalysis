# Analysis of My Feature Map

## Prerequisite

**IMPORTANT !!**

Before you run the training code below, make sure you have install Tensorflow V2 and above and Segmentation_models library. To install, run the command below

```bash
pip install segmentation-models
```

## Instruction to train all models

### FCN

1. `python train.py fcn yourpath`, replace yourpath to seagull datasetpath

### UNET

1. `python train.py fcn yourpath`, replace yourpath to seagull datasetpath

### FPN

1. `python train.py fcn yourpath`, replace yourpath to seagull datasetpath

## Experimental results

### UAvid Dataset

Uses dice loss and with pixelwise crossentropy

| MODEL                      | BACKBONE       | LR   | Test IoU | Train IoU |
| -------------------------- | -------------- | ---- | -------- | --------- |
| FPN                        | efficientNetB0 | 1e-5 | 0.476    | 0.604     |
| UNET                       | efficientNetB0 | 1e-5 | 0.405    | 0.516     |
| FCN                        | efficientNetB0 | 1e-5 | 0.192    | 0.258     |
| UNET + FPN (PRODUCT)       | efficientNetB0 | 1e-5 | 0.566    | 0.639     |
| UNET + FPN (SUM)           | efficientNetB0 | 1e-5 | 0.523    | 0.589     |
| UNET + FPN (CONCATENATION) | efficientNetB0 | 1e-5 | 0.574    | 0.646     |
| FCN + FPN (CONCATENATION)  | efficientNetB0 | 1e-5 | 0.446    | 0.394     |

### Seagull Dataset

| MODEL         | BACKBONE       | LR   | ALPHA (IOU)    | RESULT (mIOU) | RESULT (loss) |
| ------------- | -------------- | ---- | -------------- | ------------- | ------------- |
| FPN (20EPOCH) | efficientNetB0 | 1e-5 | [0.999, 0.001] | 0.79392844    |
| FPN (40EPOCH) | efficientNetB0 | 1e-5 | [0.999, 0.001] | 0.817993      | 0.6183606     |
| UNET          | efficientNetB0 | 1e-6 | [0.999, 0.001] |
| FCN           | efficientNetB0 | 1e-4 | [0.999, 0.001] |

```

```
