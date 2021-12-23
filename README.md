# Analysis of My Feature Map

## UAvid Dataset

### Description

Uses dice loss and with pixelwise crossentropy

| MODEL | BACKBONE       | LR    | ALPHA (IOU) | RESULT (mIOU) | RESULT (loss) |
| ----- | -------------- | ----- | ----------- | ------------- | ------------- |
| FPN   | efficientNetB0 | 1e-5  | 0.5         |
| UNET  | resnet50       | 1e-4  | 0.5         |
| FCN   | efficientNetB0 | 2e-05 | 0.5         |

## Seagull Dataset

| MODEL         | BACKBONE       | LR   | ALPHA (IOU)    | RESULT (mIOU) | RESULT (loss) |
| ------------- | -------------- | ---- | -------------- | ------------- | ------------- |
| FPN (20EPOCH) | efficientNetB0 | 1e-5 | [0.999, 0.001] | 0.79392844    |
| FPN (40EPOCH) | efficientNetB0 | 1e-5 | [0.999, 0.001] | 0.817993      | 0.6183606     |
| UNET          | efficientNetB0 | 1e-6 | [0.999, 0.001] |
| FCN           | efficientNetB0 | 1e-4 | [0.999, 0.001] |
