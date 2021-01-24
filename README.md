# Hack Cambridge 2021

This repo contains our team's submission to the Huawei Challenge on Hack Cambridge 2021 Hackathon ([Problem statement](https://hackcambridge.com/dashboard/challenges)).

## Motive
To utilize Huawei's Mindspore package for performing image classification on edge devices,i.e, devices with minimal computation power.

## What we used
- CIFAR-10 dataset 
- Resnet 18
- Quantization Aware Training
- Image preprocessing and parallelism achieved by Mindspore package

## Files description
- [preprocess.py](preprocess.py) <br>
    Applies image preprocessing operations from mindspore library <br>
- [resnetv2.py](resnetv2.py) <br>
    Defines implementation of Resnet architecture <br>
- [train.py](train.py) <br>
    Driver code for training the model <br>
- [eval.py](eval.py) <br>
    Evaluation script for evaluating an particular model checkpoint <br>
    
## Model Details
- Resnet-18 architecture model was trained using Quantization-Aware-Training(QAT) method. 
- Adam optimizer and SoftmaxCrossEntropyWithLogits loss function were used. 
- The model was trained for a total of 40 epochs on a NVIDIA-1070 GPU using mindspore-ai backend. 

## Recreation
**Steps for training the model:**
```bash
$python train.py

usage: train.py [-h] [-bsize BATCHSIZE] [-repeatNum REPEATNUM] [-dir SAVEDIR]
               [-e EPOCH] [-opt OPTIMIZER] [-lr LEARNINGRATE] [-m MOMENTUM]
               [-wDecay WEIGHTDECAY]

optional arguments:
  -h, --help            show this help message and exit
  -bsize BATCHSIZE, --batchsize BATCHSIZE
                        batch size
  -repeatNum REPEATNUM, --repeatNum REPEATNUM
                        repeat num
  -dir SAVEDIR, --savedir SAVEDIR
                        save directory
  -e EPOCH, --epoch EPOCH
                        no of epochs
  -opt OPTIMIZER, --optimizer OPTIMIZER
                        optimizer
  -lr LEARNINGRATE, --learningRate LEARNINGRATE
                        learning rate
  -m MOMENTUM, --momentum MOMENTUM
                        momentum
  -wDecay WEIGHTDECAY, --weightDecay WEIGHTDECAY
                        optimizer

```

**Steps for evaulating the model:**
```bash
$python eval.py

usage: eval.py [-h] [-loc CHECK_POINT]

optional arguments:
  -h, --help            show this help message and exit
  -loc CHECK_POINT, --check_point CHECK_POINT
                        Model checkpoint file


```
    
## Results

| Model Architecture  | Accuracy |
| ------------- | ------------- |
| Resnet-18  | 87.86  |

## Contributions

- Submitted a [pull request](https://gitee.com/mindspore/mindspore/pulls/11587) to Mindspore on implemented Resnet-18 Architecture

## Future Scope

- Implementing Adversarial Training to make the model more robust to real-time data.

