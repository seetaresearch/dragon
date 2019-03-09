# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

import os
import sys
import time
import shutil
import argparse
import cv2

from dragon.tools.db import LMDB
from dragon.vm.caffe.proto import caffe_pb2


def resize_image(im, resize):
    """Resize the image by the shortest edge.

    Parameters
    ----------
    im : numpy.ndarray
        The image.
    resize : int
        The size of the shortest edge.

    Returns
    -------
    numpy.ndarray
        The resized image.

    """
    if im.shape[0] > im.shape[1]:
        newsize = (resize, im.shape[0] * resize / im.shape[1])
    else:
        newsize = (im.shape[1] * resize / im.shape[0], resize)
    im = cv2.resize(im, newsize)
    return im


def make_db(args):
    """Make the sequential database for images.

    Parameters
    ----------
    database : str
        The path of database.
    root : str
        The root folder of raw images.
    list : str
        The path of image list file.
    resize : int
        The size of the shortest edge. Default is ``0`` (Disabled).
    zfill : int
        The number of zeros for encoding keys.
    quality : int
        JPEG quality for encoding, 1-100. Default is ``95``.
    shuffle : boolean
        Whether to randomize the order in list file.

    """
    if os.path.isfile(args.list) is False:
        raise ValueError('the path of image list is invalid.')
    if os.path.isdir(args.database) is True:
        raise ValueError('the database is already exist or invalid.')

    print('start time: ', time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime()))

    db = LMDB(max_commit=10000)
    db.open(args.database, mode='w')

    total_line = sum(1 for line in open(args.list))
    count = 0
    zfill_flag = '{0:0%d}' % (args.zfill)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), args.quality]

    start_time = time.time()

    with open(args.list, 'r') as input_file:
        records = input_file.readlines()
        if args.shuffle:
            import random
            random.shuffle(records)

        for record in records:
            count += 1
            if count % 10000 == 0:
                now_time = time.time()
                print('{0} / {1} in {2:.2f} sec'.format(
                    count, total_line, now_time - start_time))
                db.commit()

            record = record.split()
            path = record[0]
            label = record[1]

            img = cv2.imread(os.path.join(args.root, path))
            if args.resize > 0:
                img = resize_image(img, args.resize)
            result, imgencode = cv2.imencode('.jpg', img, encode_param)

            datum = caffe_pb2.Datum()
            datum.height, datum.width, datum.channels = img.shape
            datum.label = int(label)
            datum.encoded = True
            datum.data = imgencode.tostring()
            db.put(zfill_flag.format(count - 1), datum.SerializeToString())

    now_time = time.time()
    print('{0} / {1} in {2:.2f} sec'.format(count, total_line, now_time - start_time))
    db.commit()
    db.close()

    # Compress the empty space
    db.open(args.database, mode='w')
    db.commit()

    shutil.copy(args.list, args.database + '/image_list.txt')
    end_time = time.time()
    print('{0} images have been stored in the database.'.format(total_line))
    print('This task finishes within {0:.2f} seconds.'.format(end_time - start_time))
    print('The size of database is {0} MB.'.
          format(float(os.path.getsize(args.database + '/data.mdb') / 1000 / 1000)))


def parse_args():
    parser = argparse.ArgumentParser(description='Create LMDB from images for classification.')
    parser.add_argument('--root', help='The root folder of raw images.')
    parser.add_argument('--list', help='The path of image list file.')
    parser.add_argument('--database', help='The path of database.')
    parser.add_argument('--zfill', type=int, default=8, help='The number of zeros for encoding keys.')
    parser.add_argument('--resize', type=int, default=0, help='The size of the shortest edge.')
    parser.add_argument('--quality', type=int, default=95, help='JPEG quality for encoding, 1-100.')
    parser.add_argument('--shuffle', type=bool, default=True, help='Whether to randomize the order in list file.')

    if len(sys.argv) < 4:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    make_db(args)