import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset as ds
from mindspore import dtype as mstype


def create_dataset(repeat_num=1, training=True, batch_size=64):
    """
    create data for next use such as training or inferring

    """
    
    if training == True:
        usage = 'train'
    else:
        usage = 'test'
    
    
    #importing the cifar10 dataset and assigning number of parallel workers as 8
    cifar_ds = ds.Cifar10Dataset('dataset/Cifar10Data/cifar-10-batches-bin/', usage = usage, num_parallel_workers = 8)

    resize_height = 224
    resize_width = 224
    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations
    random_crop_op = C.RandomCrop((32, 32), (4, 4, 4, 4)) # padding_mode default CONSTANT
    random_horizontal_op = C.RandomHorizontalFlip() # flips the image horizontally 
    resize_op = C.Resize((resize_height, resize_width)) # interpolation default BILINEAR
    rescale_op = C.Rescale(rescale, shift) # rescales the input image
    normalize_op = C.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # normalize the array with given mean and standard deviation
    changeswap_op = C.HWC2CHW() # transforms H,W,C to C,H,W
    type_cast_op = C2.TypeCast(mstype.int32)

    c_trans = []
    if training:
        c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op,
                changeswap_op]

    # apply map operations on images
    cifar_ds = cifar_ds.map(operations=type_cast_op, input_columns="label")
    cifar_ds = cifar_ds.map(operations=c_trans, input_columns="image")

    # apply shuffle operations
    cifar_ds = cifar_ds.shuffle(buffer_size=10)

    # apply batch operations
    cifar_ds = cifar_ds.batch(batch_size=batch_size, drop_remainder=True)

    # apply repeat operations
    cifar_ds = cifar_ds.repeat(repeat_num)

    return cifar_ds
