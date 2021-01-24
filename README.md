# Hex Cam 2021

This repo contains submission to Huawei Challenge  ([problem statement](https://hackcambridge.com/dashboard/challenges)).

## Motive
To use Huawei's Mindspore package for image classification on edge devices with minimal computation power.

## What we did
- used CIFAR-10 dataset 
- used quantization while training

## Files description
preprocess.py <br>
    Applies image preprocessing operations from mindspore library <br>
resnetv2.py <br>
    Defines implementation of Resnet architecture <br>
train.py <br>
    Driver code for training the model <br>
eval.py <br>
    Evaluation script for evaluating an particular model checkpoint <br>




