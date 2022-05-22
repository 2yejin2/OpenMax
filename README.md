## Towards Open Set Deep Networks

This repository contains the PyTorch implementation of OpenMax. As for the details of the paper, please refer to [here](https://arxiv.org/abs/1511.06233)
.

### Requirements
* ubuntu 18.0.4, cuda >= 10.2
* python >= 3.6.8
* torch >= 1.2.0
* torchvision >= 0.4.0 

### Usage

* Train a classifier with the unknown detection benchmark

``` 
  CUDA_VISIBLE_DEVICES=gpu-ids python main.py
```


* Test the trained classifier with the unknown detection benchmark

```
  CUDA_VISIBLE_DEVICES=gpu-ids python test.py
```

