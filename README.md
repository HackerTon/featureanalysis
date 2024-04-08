# Semantatic segmentation trainer
![screenshot](src/notebook/test.png)

Heart and lung segmentation with bounding box visualisation


## How to train
Run `python main -p {dataset} -b {batch size} -x {experiment number} -m {compute mode} -l {learning rate}`

- {dataset}: path to your dataset 
- {batch size}: batch size 
- {experiment number}: experiment number (Refer to [trainer.py](src/service/trainer.py))  

Example command 
`python main -p data/textocr -b 32 -x 0 -m mps -l 0.001`

## How to monitor training progress
Run `tensorboard --logdir data/log`

## What have I did
### Multinet
1. Increase lateral channel size to half of final backbone channel output **(2048 / 4 = 512)**