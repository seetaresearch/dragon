# --------------------------------------------------------
# Cifar-10 for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

""" Generate database """

import os
import sys
import time
import shutil
import tarfile
from six.moves import range as xrange

import cv2

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
    extract_path = 'data/extract'
    if not os.path.exists(os.path.join(extract_path, 'JPEGImages')):
        os.makedirs(os.path.join(extract_path, 'JPEGImages'))
    if not os.path.exists(os.path.join(extract_path, 'ImageSets')):
        os.makedirs(os.path.join(extract_path, 'ImageSets'))

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
                filename = str(total_idx).zfill(ZFILL) + '.jpg'
                cv2.imwrite(os.path.join(extract_path, 'JPEGImages', filename), im)
                images_list.append((filename, str(label)))
                total_idx += 1

    # make list
    with open(os.path.join(extract_path, 'ImageSets', 'train.txt'), 'w') as f:
        for i in xrange(50000):
            item = images_list[i][0] + ' ' + images_list[i][1]
            if i != 49999: item += '\n'
            f.write(item)

    with open(os.path.join(extract_path, 'ImageSets', 'test.txt'), 'w') as f:
        for i in xrange(50000, 60000):
            item = images_list[i][0] + ' ' + images_list[i][1]
            if i != 59999: item += '\n'
            f.write(item)


def make_db(image_path, label_path, database_path):
    if os.path.isfile(label_path) is False:
        raise ValueError('input path is empty or wrong.')
    if os.path.isdir(database_path) is True:
        raise ValueError('the database path is already exist.')

    print('start time: ', time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime()))

    db = LMDB(max_commit=10000)
    db.open(database_path, mode='w')

    total_line = sum(1 for line in open(label_path))
    count = 0
    zfill_flag = '{0:0%d}' % (ZFILL)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]

    start_time = time.time()

    with open(label_path, 'r') as input_file:
        for record in input_file:
            count += 1
            if count % 10000 == 0:
                now_time = time.time()
                print('{0} / {1} in {2:.2f} sec'.format(
                    count, total_line, now_time - start_time))
                db.commit()

            record = record.split()
            path = record[0]
            label = record[1]

            img = cv2.imread(os.path.join(image_path ,path))
            result, imgencode = cv2.imencode('.jpg', img, encode_param)

            datum = caffe_pb2.Datum()
            datum.height, datum.width, datum.channels = img.shape
            datum.label = int(label)
            datum.encoded = True
            datum.data = imgencode.tostring()
            db.put(zfill_flag.format(count - 1), datum.SerializeToString())

    now_time = time.time()
    print('{0} / {1} in {2:.2f} sec'.format(count, total_line, now_time - start_time))
    db.put('size', wrapper_str(str(count)))
    db.put('zfill', wrapper_str(str(ZFILL)))
    db.commit()
    db.close()

    shutil.copy(label_path, database_path + '/image_list.txt')
    end_time = time.time()
    print('{0} images have been stored in the database.'.format(total_line))
    print('This task finishes within {0:.2f} seconds.'.format(
        end_time - start_time))
    print('The size of database is {0} MB.'.format(
        float(os.path.getsize(database_path + '/data.mdb') / 1000 / 1000)))


if __name__ == '__main__':

    untar('data/cifar-10-python.tar.gz')

    extract_images()

    make_db('data/extract/JPEGImages',
            'data/extract/ImageSets/train.txt',
            'data/train_lmdb')

    make_db('data/extract/JPEGImages',
            'data/extract/ImageSets/test.txt',
            'data/test_lmdb')
