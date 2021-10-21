# Analysis of My Feature Map

## UAvid Dataset

### Description

Uses dice loss and with pixelwise crossentropy

| MODEL | BACKBONE       | LR    | ALPHA (IOU) | RESULT (mIOU) | RESULT (loss) |
| ----- | -------------- | ----- | ----------- | ------------- | ------------- |
| FPN   | efficientNetB0 | 1e-5  | 0.5         |
| FPN   | efficientNetB0 | 5e-5  | 1.0         |
| UNET  | resnet50       | 1e-4  | 0.5         |
| FCN   | efficientNetB0 | 2e-05 | 0.5         |

## Seagull Dataset

| MODEL | BACKBONE       | LR   | ALPHA (IOU) | RESULT (mIOU) | RESULT (loss) |
| ----- | -------------- | ---- | ----------- | ------------- | ------------- |
| FPN   | efficientNetB0 | 1e-5 | 0.5         |
| UNET  | resnet50       |      |             |
| FCN   | efficientNetB0 |      |             |
