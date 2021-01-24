from resnetv2 import resnet18
from preprocess import create_dataset
from mindspore.compression.quant import QuantizationAwareTraining
from mindspore.nn.optim import Adam
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore import load_checkpoint
from mindspore import Model, context

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")


num_classes = 10
lr = 0.01
weight_decay = 1e-4

resnet = resnet18(num_classes)

ls = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
opt = Adam(resnet.trainable_params(),lr, weight_decay)

quantizer = QuantizationAwareTraining(bn_fold=False)
quant = quantizer.quantize(resnet)

load_checkpoint("./new_check/train_resnet_cifar10-41_10.ckpt", net=quant) # loading the custom trained checkpoint

eval_data = create_dataset(training = False) # define the test dataset


model = Model(quant, loss_fn=ls, optimizer=opt, metrics={'acc'})

acc = model.eval(eval_data)

print('Accuracy of model is: ', acc['acc'] * 100)
