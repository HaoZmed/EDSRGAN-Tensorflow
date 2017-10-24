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
        
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
    
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

            input_image_list.append(os.path.join(self.input_data_dir, in_file))
            output_image_list.append(os.path.join(self.output_data_dir, out_file))
        if self.shuffle:
            combined = list(zip(input_image_list, output_image_list))
            prng = np.random.RandomState(self.random_state)
            prng.shuffle(combined)
            input_image_list, output_image_list = zip(*combined)

        return input_image_list, output_image_list
    
    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

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
            in_image = imread(input_image_list[i]).astype('int32')
            out_image = imread(output_image_list[i]).astype('int32')
            # If the image is in grayscale, resize to (h, w, 1)
            if len(in_image.shape) < 3:
                in_image = in_image.reshape(in_image.shape+(1,))
                out_image = out_image.reshape(out_image.shape+(1,))
            in_shape = np.array(in_image.shape, np.int32)
            out_shape = np.array(out_image.shape, np.int32)
            sample = tf.train.Example(features=tf.train.Features(feature={
                'in_shape': self._bytes_feature(in_shape.tostring()),
                'out_shape': self._bytes_feature(out_shape.tostring()),
                'in_image_raw': self._bytes_feature(in_image.tostring()),
                'out_image_raw': self._bytes_feature(out_image.tostring())
            }))
            writer.write(sample.SerializeToString())
        writer.close()
        print('Images are transfered and stored as {}'.format(filename))
