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

| MODEL | BACKBONE       | LR   | Test IoU | Train IoU |
| ----- | -------------- | ---- | -------- | --------- |
| FPN   | efficientNetB0 | 1e-5 | 0.519    | 0.615     |
| UNET  | efficientNetB0 | 1e-5 | OTW      | OTW       |
| FCN   | efficientNetB0 | 1e-5 | 0.248    | 0.257     |

### Seagull Dataset

| MODEL         | BACKBONE       | LR   | ALPHA (IOU)    | RESULT (mIOU) | RESULT (loss) |
| ------------- | -------------- | ---- | -------------- | ------------- | ------------- |
| FPN (20EPOCH) | efficientNetB0 | 1e-5 | [0.999, 0.001] | 0.79392844    |
| FPN (40EPOCH) | efficientNetB0 | 1e-5 | [0.999, 0.001] | 0.817993      | 0.6183606     |
| UNET          | efficientNetB0 | 1e-6 | [0.999, 0.001] |
| FCN           | efficientNetB0 | 1e-4 | [0.999, 0.001] |

```

```
