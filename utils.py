"""
Utilities:
- data transfer/read
- batch generations
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
from skimage.io import imread


# DATA RELATED UTILS
class Data(object):
    """Defines the Data class to manipulate the data used in the model.
        The input/output image should be stored in different paths
        - path_to_data
        |   - input
            |   - 1.png
            |   - 2.png
                ...
        |   - ouput
            |   - 1.png
            |   - 2.png 
                ...
        I transfer the image into grayscale because I am mostly dealing 
        with medical images in grayscale. The image has the dimension of
        [height, width, 1] and the code works for color image as well. 
        The images could have DIFFERENT sizes but the low-resolution and
        its corresponding high-resolution images should be matched.
    """

    def __init__(self, input_data_dir, save_dir, name, mode, output_data_dir='', shuffle=True, random_state=99):
        self.input_data_dir = input_data_dir
        self.output_data_dir = output_data_dir
        self.save_dir = save_dir
        self.name = name
        self.shuffle = shuffle
        self.random_state = random_state
        self.mode = mode

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def get_files(self):
        """Get the list of input images and output images in pairs
        Args:
            None
        Returns:
            input_image_list, output_image_list
        """

        if self.mode == 'train' or self.mode == 'valid':

            input_image_list = list()
            output_image_list = list()

            assert (len(os.listdir(self.input_data_dir)) ==
                    len(os.listdir(self.output_data_dir))), \
                'The number of input and output files are different'

            for in_file, out_file in zip(os.listdir(self.input_data_dir),
                                         os.listdir(self.output_data_dir)):
                assert (in_file.endswith('.png')), 'The image should be *.png'
                assert (out_file.endswith('.png')), 'The image should be *.png'

                input_image_list.append(
                    os.path.join(self.input_data_dir, in_file))
                output_image_list.append(os.path.join(
                    self.output_data_dir, out_file))
            if self.shuffle:
                combined = list(zip(input_image_list, output_image_list))
                prng = np.random.RandomState(self.random_state)
                prng.shuffle(combined)
                input_image_list, output_image_list = zip(*combined)

            return input_image_list, output_image_list

        else:

            input_image_list = list()
            output_image_list = list()

            for in_file in os.listdir(self.input_data_dir):
                assert (in_file.endswith('.png')), 'The image should be *.png'
                input_image_list.append(
                    os.path.join(self.input_data_dir, in_file))

            return input_image_list, output_image_list

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def convert_to_tfrecord(self, input_image_list, output_image_list, num_files=1):
        """Convert all images pairs to tfrecord files and save as name
        Args:
            input_image_list, output_image_list: filenames of the images
        Return:
            None
        """

        i = 0

        if self.mode == 'train' or self.mode == 'valid':
            for index_file in range(num_files):
                filename = os.path.join(
                    self.save_dir, self.name + '_' + str(index_file + 1) + '.tfrecords')
                print('Writing {}'.format(filename))
                writer = tf.python_io.TFRecordWriter(filename)
                for _ in range(int(len(input_image_list) / num_files)):
                    if i >= len(input_image_list):
                        break
                    in_image = imread(input_image_list[i]).astype('int32')
                    out_image = imread(output_image_list[i]).astype('int32')
                    # If the image is in grayscale, resize to (h, w, 1)
                    if len(in_image.shape) < 3:
                        in_image = in_image.reshape(in_image.shape + (1,))
                        out_image = out_image.reshape(out_image.shape + (1,))
                    in_shape = np.array(in_image.shape, np.int32)
                    out_shape = np.array(out_image.shape, np.int32)
                    sample = tf.train.Example(features=tf.train.Features(feature={
                        'in_shape': self._bytes_feature(in_shape.tostring()),
                        'out_shape': self._bytes_feature(out_shape.tostring()),
                        'in_image_raw': self._bytes_feature(in_image.tostring()),
                        'out_image_raw': self._bytes_feature(out_image.tostring())
                    }))
                    writer.write(sample.SerializeToString())
                    i += 1
                writer.close()
        else:
            for index_file in range(num_files):
                filename = os.path.join(
                    self.save_dir, self.name + str(index_file + 1) + '.tfrecords')
                print('Writing {}'.format(filename))
                writer = tf.python_io.TFRecordWriter(filename)
                for _ in range(int(len(input_image_list) / num_files)):
                    if i >= len(input_image_list):
                        break
                    in_image = imread(input_image_list[i]).astype('int32')
                    # If the image is in grayscale, resize to (h, w, 1)
                    if len(in_image.shape) < 3:
                        in_image = in_image.reshape(in_image.shape + (1,))
                    in_shape = np.array(in_image.shape, np.int32)
                    sample = tf.train.Example(features=tf.train.Features(feature={
                        'in_shape': self._bytes_feature(in_shape.tostring()),
                        'in_image_raw': self._bytes_feature(in_image.tostring())
                    }))
                    writer.write(sample.SerializeToString())
                    i += 1
                writer.close()
        print('Images are transfered and stored as {}'.format(filename))


class Preprocessor(object):

    def __init__(self, crop_size_in, crop_size_out, channel_in):
        self.crop_size_in = crop_size_in
        self.crop_size_out = crop_size_out
        self.channel_in = channel_in

    def _parse_function(self, example_proto, max=255):
        features = {'in_shape': tf.FixedLenFeature([], tf.string),
                    'out_shape': tf.FixedLenFeature([], tf.string),
                    'in_image_raw': tf.FixedLenFeature([], tf.string),
                    'out_image_raw': tf.FixedLenFeature([], tf.string)}
        parsed_features = tf.parse_single_example(example_proto, features)
        in_shape = tf.decode_raw(parsed_features["in_shape"], tf.int32)
        in_image = tf.decode_raw(parsed_features["in_image_raw"], tf.int32)
        out_shape = tf.decode_raw(parsed_features["out_shape"], tf.int32)
        out_image = tf.decode_raw(parsed_features["out_image_raw"], tf.int32)

        in_image = tf.cast(tf.reshape(in_image, in_shape), tf.float32)
        out_image = tf.cast(tf.reshape(out_image, out_shape), tf.float32)

        in_image = in_image / max * 2. - 1.
        out_image = out_image / max * 2. - 1.

        return in_image, out_image

    def _preprocessing_function(self, image_in, image_out):

        in_shape = tf.shape(image_in)
        h_offset = tf.random_uniform(
            [], minval=0, maxval=in_shape[0] - self.crop_size_in, dtype=tf.int32)
        w_offset = tf.random_uniform(
            [], minval=0, maxval=in_shape[1] - self.crop_size_in, dtype=tf.int32)
        box_start = tf.stack([h_offset, w_offset, tf.constant(0)])
        box_size = tf.constant(
            (self.crop_size_in, self.crop_size_in, self.channel_in))
        cropped_image_in = tf.slice(image_in, box_start, box_size)

        out_shape = tf.shape(image_out)
        h_offset = tf.scalar_mul(4, h_offset)
        w_offset = tf.scalar_mul(4, w_offset)
        box_start = tf.stack([h_offset, w_offset, tf.constant(0)])
        box_size = tf.constant(
            (self.crop_size_out, self.crop_size_out, self.channel_in))
        cropped_image_out = tf.slice(image_out, box_start, box_size)

        return cropped_image_in, cropped_image_out
