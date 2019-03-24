import pandas as pd
import numpy as np
import h5py


def flatBatch2Tensor(batchData, imSize, channels):
    splitByChannel = [
        batchData[:, (chan * imSize ** 2):((chan + 1) * imSize ** 2)].reshape((-1, imSize, imSize, 1)) \
        for chan in range(channels)]
    tensorBatchData = np.concatenate(splitByChannel, 3)

    return tensorBatchData


def flatTensor2Batch(Tensor, imSize, channels):
    preserveByChannel = [
        Tensor[:, :, :, chan:(chan+1)].reshape([-1, imSize ** 2])
        for chan in range(channels)]
    batchData = np.concatenate(preserveByChannel, 1)
    return batchData

SCRATCH = set(['ACTIN', 'BUDNECK', 'BUDTIP', 'CELLPERIPHERY', 'CYTOPLASM', 'ENDOSOME', 'ER'])
TRANS = set(['GOLGI', 'MITOCHONDRIA', 'NUCLEARPERIPHERY', 'NUCLEOLUS', 'NUCLEI', 'PEROXISOME', 'SPINDLEPOLE'])


def get_hdf5_training_data():
    f_train = h5py.File('datasets/Chong_train_set.hdf5', 'r')
    names = f_train['label_names']
    index_to_be_delete = [i for i, v in enumerate(names) if bytes.decode(v) not in SCRATCH]
    imgs = np.array(f_train['data1'])
    labels = np.array(f_train['Index1'])
    imgs = np.delete(imgs, index_to_be_delete, axis=0)
    labels = np.delete(labels, index_to_be_delete, axis=0)
    f_train.close()
    return imgs, labels


def get_hdf5_testing_data():
    f_test = h5py.File('datasets/Chong_test_set.hdf5', 'r')
    names = f_test['label_names']
    index_to_be_delete = [i for i, v in enumerate(names) if bytes.decode(v) not in SCRATCH]
    imgs = np.array(f_test['data1'])
    labels = np.array(f_test['Index1'])
    imgs = np.delete(imgs, index_to_be_delete, axis=0)
    labels = np.delete(labels, index_to_be_delete, axis=0)
    f_test.close()
    return imgs, labels
