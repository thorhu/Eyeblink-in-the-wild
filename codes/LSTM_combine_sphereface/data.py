import tensorflow as tf
import os
import h5py
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"]='1'


def data_input(datapath):

    # load the data
    #datafile = datapath
    data_mat = h5py.File(datapath)
    images = data_mat['data'][:].transpose((0, 2, 1))
    #labels = data_mat['label'][:].transpose((2,1,0))
    sign = data_mat['label'][:].transpose((1, 0))
    return images, sign


def test_input(datapath, _batch_size):
    # load data
    data_mat = h5py.File(datapath)
    images_ = data_mat['data'][:].transpose((0, 2, 1))
    sign_ = data_mat['label'][:].transpose((1, 0))
    images=np.asarray([images_[i] for i in range(_batch_size)])
    sign = np.asarray([int(sign_[i]) for i in range(_batch_size)])
    return images, sign
