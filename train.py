# import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from models import *
from skimage.io import imread, imsave
import tensorflow as tf
import utils as utils
import shutil

# import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt

# Test loop for 2 epochs
CROP_SIZE_IN = 48
CROP_SIZE_OUT = CROP_SIZE_IN * 4
CHANNEL_IN = 1
LOGDIR = './checkpoints/experiment_1/'

if os.path.exists(LOGDIR):
    shutil.rmtree(LOGDIR, ignore_errors=True)
    # os.mkdir(LOGDIR)

N_EPOCH = 100

PATH_TO_TFRECORDS = 'D:/PROJECT_DATA/axial_superresolution//tfrecords/'

with tf.name_scope('data_pipeline'):
    filenames_train = [os.path.join(PATH_TO_TFRECORDS, f) for f in
                       os.listdir(PATH_TO_TFRECORDS) if 'train' in f]
    train_dataset = tf.contrib.data.TFRecordDataset(filenames_train)

    preprocessor = utils.Preprocessor(crop_size_in=CROP_SIZE_IN,
                                      crop_size_out=CROP_SIZE_OUT, channel_in=CHANNEL_IN)
    train_dataset = train_dataset.map(preprocessor._parse_function)
    train_dataset = train_dataset.map(preprocessor._preprocessing_function)
    train_dataset = train_dataset.batch(1)

    iterator = train_dataset.make_initializable_iterator()

    input_images, output_images = iterator.get_next()


srnet = EDSR(n_blocks=16)

predicted_images = srnet.generator(input_images)

tf.summary.image('input_image', input_images, max_outputs=4)
tf.summary.image('output_image', output_images, max_outputs=4)
tf.summary.image('predicted_image', predicted_images, max_outputs=4)

with tf.name_scope('loss'):
    loss_l1 = tf.reduce_mean(
        tf.abs(tf.subtract(predicted_images, output_images)))
    loss = loss_l1  # Other loss to be added
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

summ = tf.summary.merge_all()
writer = tf.summary.FileWriter(LOGDIR, graph=sess.graph)
saver = tf.train.Saver()


# Training the network
print('Starting training ...')
i = 1
for epoch in range(N_EPOCH):
    sess.run(iterator.initializer)
    while True:
        try:
            sess.run(train_op)
            if i % 100 == 0:
                [train_loss, s] = sess.run([loss, summ])
                writer.add_summary(s, i)
            i += 1
        except tf.errors.OutOfRangeError:
            break

    if epoch % 5 == 0:
        saver.save(sess, os.path.join(LOGDIR, 'model.ckpt'), i)