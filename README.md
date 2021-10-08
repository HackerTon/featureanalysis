# Analysis of My Feature Map

| MODEL | BACKBONE       | LR    | ALPHA (IOU) | RESULT (mIOU) | RESULT (loss) |
| ----- | -------------- | ----- | ----------- | ------------- | ------------- |
| FPN   | efficientNetB0 | 1e-5  | 0.5         |
| FPN   | efficientNetB0 | 5e-5  | 1.0         |
| UNET  | resnet50       | 1e-4  | 0.5         |
| FCN   | efficientNetB0 | 2e-05 | 0.5         |
