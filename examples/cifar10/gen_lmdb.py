# --------------------------------------------------------
# Cifar-10 for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

""" Generate database """

import os
import sys
import time
import tarfile
import numpy as np
from six.moves import range as xrange

from dragon.tools.db import LMDB
from dragon.vm.caffe.proto import caffe_pb2

ZFILL = 8

def untar(tar_file):
    t = tarfile.open(tar_file)
    t.extractall(path='data')

def wrapper_str(raw_str):
    if sys.version_info >= (3, 0):
        return raw_str.encode()
    return raw_str

def extract_images():
    prefix = 'data/cifar-10-batches-py'
    batches = [os.path.join(prefix, 'data_batch_{}'.format(i)) for i in xrange(1, 6)]
    batches += [os.path.join(prefix, 'test_batch')]

    total_idx = 0
    images_list = []

    # process batches
    for batch in batches:
        with open(batch, 'rb') as f:
            if sys.version_info >= (3, 0):
                import pickle
                with open(batch, 'rb') as f:
                    dict = pickle.load(f, encoding='bytes')
            else:
                import cPickle
                with open(batch, 'rb') as f:
                    dict = cPickle.load(f)
            for item_idx in xrange(len(dict[wrapper_str('labels')])):
                im = dict[wrapper_str('data')][item_idx].reshape((3, 32, 32))
                label = dict[wrapper_str('labels')][item_idx]
                im = im.transpose((1, 2, 0))
                im = im[:, :, ::-1]
                images_list.append((im, str(label)))
                total_idx += 1

    return images_list


def make_db(images_list, database_path):
    if os.path.isdir(database_path) is True:
        raise ValueError('the database path is already exist.')

    print('start time: ', time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime()))

    db = LMDB(max_commit=10000)
    db.open(database_path, mode='w')

    total_line = len(images_list)
    count = 0
    zfill_flag = '{0:0%d}' % (ZFILL)

    start_time = time.time()

    for record in images_list:
        count += 1
        if count % 10000 == 0:
            now_time = time.time()
            print('{0} / {1} in {2:.2f} sec'.format(
                count, total_line, now_time - start_time))
            db.commit()

        img = record[0]
        label = record[1]

        datum = caffe_pb2.Datum()
        datum.height, datum.width, datum.channels = img.shape
        datum.label = int(label)
        datum.encoded = False
        datum.data = img.tostring()
        db.put(zfill_flag.format(count - 1), datum.SerializeToString())

    now_time = time.time()
    print('{0} / {1} in {2:.2f} sec'.format(count, total_line, now_time - start_time))
    db.put('size', wrapper_str(str(count)))
    db.put('zfill', wrapper_str(str(ZFILL)))
    db.commit()
    db.close()

    end_time = time.time()
    print('{0} images have been stored in the database.'.format(total_line))
    print('This task finishes within {0:.2f} seconds.'.format(
        end_time - start_time))
    print('The size of database is {0} MB.'.format(
        float(os.path.getsize(database_path + '/data.mdb') / 1000 / 1000)))


if __name__ == '__main__':

    untar('data/cifar-10-python.tar.gz')

    images_list = extract_images()

    make_db(images_list[0:50000], 'data/train_lmdb')

    make_db(images_list[50000:60000], 'data/test_lmdb')
