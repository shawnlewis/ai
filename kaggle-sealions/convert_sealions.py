from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf
import pandas as pd
from PIL import Image
import numpy as np

from datasets import dataset_utils

_NUM_VALIDATION = 4000

_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 5


def decode_jpeg(fname):
    im = Image.open(fname)
    rgb_im = im.convert(mode='RGB')
    np_im = np.array(rgb_im)
    return np_im, np_im.shape[0], np_im.shape[1]

def read_labels(dataset_dir):
    return pd.read_csv(os.path.join(dataset_dir, 'train.csv'))

def get_filenames(dataset_dir,
        test_or_train='train', jpg_or_tiff='tif'):
    image_dir = os.path.join(dataset_dir, '%s-%s' % (test_or_train, jpg_or_tiff))
    filenames = [os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)]
    return filenames

def get_classnames(labels_csv):
    classnames = set()
    for tags in labels_csv.tags:
        classnames.update(tags.split())
    return sorted(classnames)

def get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = 'sealions_%s_%05d-of-%05d.tfrecord' % (
          split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)


def convert_dataset(split_name, filenames, fnames_to_class_ids, output_dir):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = get_dataset_filename(output_dir, split_name,shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(filenames), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_data, height, width = decode_jpeg(filenames[i])
            #image_data = tf.gfile.FastGFile(filenames[i], 'r').read()
            #height, width = image_reader.read_image_dims(sess, image_data)

            image_name = filenames[i].split('.')[0]
            class_ids = fnames_to_class_ids[os.path.basename(image_name)]

            example = dataset_utils.image_to_tfexample(
                image_data.tostring(), 'raw', height, width, class_ids)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()

def run(dataset_dir):
    """Runs the download and conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
    """
    photo_filenames = get_filenames(dataset_dir, jpg_or_tiff='jpg')
    labels_csv = read_labels(dataset_dir)
    class_names = get_classnames(labels_csv)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    output_dir = os.path.join(dataset_dir, 'tensorflow')

    random.seed(_RANDOM_SEED)
    random.shuffle(photo_filenames)
    training_filenames = photo_filenames[_NUM_VALIDATION:]
    validation_filenames = photo_filenames[:_NUM_VALIDATION]

    fnames_to_class_ids = {}
    for fname, tags in zip(labels_csv.image_name, labels_csv.tags):
        fnames_to_class_ids[fname] = [
                class_names_to_ids[i] for i in tags.split()]

    convert_dataset('train', training_filenames, fnames_to_class_ids,
            output_dir)
    convert_dataset('validation', validation_filenames, fnames_to_class_ids,
            output_dir)

    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, output_dir)

    print('\nFinished converting!')

def main(argv):
    run(argv[1])

if __name__ == '__main__':
    main(sys.argv)
