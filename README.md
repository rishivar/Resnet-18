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
    
## Results

| Model Architecture  | Accuracy |
| ------------- | ------------- |
| Resnet-18  | 87.86  |



