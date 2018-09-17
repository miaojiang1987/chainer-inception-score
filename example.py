import argparse
import numpy as np
import os
from PIL import Image
import chainer
from chainer import cuda
from chainer import datasets
from chainer import serializers
from inception_score import Inception
from inception_score import inception_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--samples', type=int, default=-1)
    parser.add_argument('--model', type=str, default='inception_score.model')
    return parser.parse_args()


def main(args):
    # Load trained model
    model = Inception()
    serializers.load_hdf5(args.model, model)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    # Load images
#    train, test = datasets.get_cifar10(ndim=3, withlabel=False, scale=255.0)
    names=os.listdir('sliced')
    first_name=names[0]
    im=Image.open('sliced/'+first_name)
    im2arr=np.array(im,dtype=np.float32)
    im2arr=im2arr[2:34,2:34,:]
    im2arr_t=np.transpose(im2arr,(2,0,1))
    
    print(im2arr_t)
    images=[im2arr_t]
    for i in range(1,60000):
        name=names[i]
        im=Image.open('sliced/'+name)
        im2arr=np.array(im,dtype=np.float32)
        im2arr=im2arr[2:34,2:34,:]
        
                     
        im2arr_t=np.transpose(im2arr,(2,0,1))
        images.append(im2arr_t)
    images_array=np.array(images)
    print(images_array.shape)
    # Use all 60000 images, unless the number of samples are specified
#    ims = np.concatenate((train, test))
#    if args.samples > 0:
#        images_array = images_array[:args.samples]

    with chainer.no_backprop_mode(), chainer.using_config('train', False):
#        mean, std = inception_score(model, ims)
         mean, std = inception_score(model, images_array)

    print('Inception score mean:', mean)
    print('Inception score std:', std)


if __name__ == '__main__':
    args = parse_args()
    main(args)
