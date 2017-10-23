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
import numpy.random as random
from skimage.io import imread


## DATA RELATED UTILS
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

    def __init__(self, input_data_dir, output_data_dir, save_dir, name, shuffle=True, random_state=99):
        self.input_data_dir = input_data_dir
        self.output_data_dir = output_data_dir
        self.save_dir = save_dir
        self.name = name
        self.shuffle = shuffle
        self.random_state = random_state
    
    def get_files(self):
        """Get the list of input images and output images in pairs
        Args:
            None
        Returns:
            input_image_list, output_image_list
        """
        
        input_image_list = list()
        output_image_list = list()

        assert (len(os.listdir(self.input_data_dir)) == 
                len(os.listdir(self.output_data_dir))), \
                'The number of input and output files are different'

        for in_file, out_file in zip(os.listdir(self.input_data_dir),
            os.listdir(self.output_data_dir)):
            assert (in_file.endswith('.png')), 'The image should be *.png'
            assert (out_file.endswith('.png')), 'The image should be *.png'

            input_image_list.append(in_file)
            output_image_list.append(out_file)

        if self.shuffle:
            combined = zip(input_image_list, output_image_list)
            prng = np.random.RandomState(self.random_state)
            prng.shuffle(combined)
            input_image_list, output_image_list = zip(*combined)

        return input_image_list, output_image_list
    
    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def convert_to_tfrecord(self, input_image_list, output_image_list):
        """Convert all images pairs to tfrecord files and save as name
        Args:
            input_image_list, output_image_list: filenames of the images
        Return:
            None
        """

        filename = os.path.join(self.save_dir, self.name+'.tfrecords')
        print('Writing {}'.format(filename))

        writer = tf.python_io.TFRecordWriter(filename)
        n_samples = len(input_image_list)
        for i in range(n_samples):
            i_image = input_image_list[i]
            o_image = output_image_list[i]
            i_height, i_width = i_image.shape[:2]
            o_height, o_width = o_image.shape[:2]
            sample = tf.train.Example(features=tf.train.Features(feature={
                'i_height': self._int64_feature(i_height),
                'i_width': self._int64_feature(i_width),
                'o_height': self._int64_feature(o_height),
                'o_width': self._int64_feature(o_width),
                'i_image_raw': self._bytes_feature(i_image.tostring()),
                'o_image_raw': self._bytes_feature(o_image.tostring())
            }))
            writer.write(sample.SerializeToString())
        writer.close()
        print('Images are transfered and stored as {}'.format(filename))
