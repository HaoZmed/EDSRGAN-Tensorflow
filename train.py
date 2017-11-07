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


# ***************************** Parameter setting *************************************** #
# Test loop for 2 epochs
CROP_SIZE_IN = 48
CROP_SIZE_OUT = CROP_SIZE_IN * 4
CHANNEL_IN = 1
EXPERIMENT_NAME = 'experiment_7'
LOGDIR = './checkpoints/'+EXPERIMENT_NAME+'/'
N_EPOCH = 100
SAVE_FREQUENCY = 5

if os.path.exists(LOGDIR):
    shutil.rmtree(LOGDIR, ignore_errors=True)

# ***************************** Data Pipeline ******************************************* #
# PATH_TO_TFRECORDS = 'D:/PROJECT_DATA/axial_superresolution//tfrecords/'
PATH_TO_TFRECORDS = '/home/hao/Downloads/MRI/'
with tf.name_scope('data_pipeline'):
    filenames_train = [os.path.join(PATH_TO_TFRECORDS, f) for f in
                    os.listdir(PATH_TO_TFRECORDS) if 'train' in f]
    train_dataset = tf.contrib.data.TFRecordDataset(filenames_train)  # TF 1.4
    preprocessor = utils.Preprocessor(crop_size_in=CROP_SIZE_IN,
                                    crop_size_out=CROP_SIZE_OUT, channel_in=CHANNEL_IN)
    train_dataset = train_dataset.map(preprocessor._parse_function)
    train_dataset = train_dataset.map(preprocessor._preprocessing_function)
    train_dataset = train_dataset.batch(1)

    iterator = train_dataset.make_initializable_iterator()
    input_images, output_images = iterator.get_next()

    # Enlarge the input image to bicubic for visualization only. The resize method in the
    # tensorboard seems to be nearest neighborhood which looks worse than bicubic method.
    input_images_bicubic = tf.image.resize_bicubic(input_images,
                                            size=(CROP_SIZE_OUT, CROP_SIZE_OUT))


# ***************************** Define train op *************************************** #
G_network = EDSR(n_blocks=8)
D_network = Discriminator()


predicted_images = G_network.generator(input_images)
_, logits_predicted = D_network.d_SMALL(predicted_images, reuse=False)
_, logits_real      = D_network.d_SMALL(output_images, reuse=True)


with tf.name_scope('loss'):
    d_loss_predicted = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_predicted,
        labels=tf.zeros_like(logits_predicted), name='d_predicted'))
    d_loss_real      = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real,
        labels=tf.ones_like(logits_real), name='d_real'))
    d_loss = d_loss_predicted + d_loss_real

    adv_loss = 1e-2*tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_predicted,
        labels=tf.ones_like(logits_predicted), name='advsarial_loss'))
    mae_loss = tf.reduce_mean(
        tf.abs(tf.subtract(predicted_images, output_images)))
    g_loss = adv_loss + mae_loss
    tf.summary.scalar('g_loss', g_loss)
    tf.summary.scalar('d_loss', d_loss)
    tf.summary.scalar('mae_loss', mae_loss)
    tf.summary.scalar('adv_loss', adv_loss)

vars_train = tf.trainable_variables()
generator_vars = [var for var in vars_train if var.name.startswith('Generator')]
discriminator_vars = [var for var in vars_train if var.name.startswith('Discriminator')]
with tf.name_scope('train'):
    train_d_op = tf.train.AdamOptimizer(
        learning_rate=0.0002, name="Adam_Discriminator").minimize(d_loss,
        var_list=discriminator_vars)
    train_g_op = tf.train.AdamOptimizer(
        learning_rate=0.0002, name="Adam_Generator").minimize(g_loss,
        var_list=generator_vars)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# ***************************** Define summarys *************************************** #
tf.summary.image('input_image', input_images_bicubic, max_outputs=100)
tf.summary.image('output_image', output_images, max_outputs=100)
tf.summary.image('predicted_image', predicted_images, max_outputs=100)
summ = tf.summary.merge_all()
writer = tf.summary.FileWriter(LOGDIR, graph=sess.graph)

input("Press 'Enter' to train")


# ************************************ Training *************************************** #
print('Starting training ...')
saver = tf.train.Saver()
i = 1
for epoch in range(N_EPOCH):
    sess.run(iterator.initializer)
    print("Epoch {}/{}".format(epoch, N_EPOCH))
    while True:
        try:
            # Update D
            sess.run(train_d_op)
            # Update G
            sess.run(train_g_op)
            if i % 100 == 0:
                s = sess.run(summ)
                writer.add_summary(s, i)
            i += 1
        except tf.errors.OutOfRangeError:
            break

    if epoch % SAVE_FREQUENCY == 0:
        saver.save(sess, os.path.join(LOGDIR, EXPERIMENT_NAME+'.ckpt'), i)