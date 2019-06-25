from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers


import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim

def vgg16():
    g = tf.Graph()
    with g.as_default():
        imgs = tf.placeholder(tf.float32, [None, 32, 32, 3])
        label = tf.placeholder(tf.float32, [None, 10])
        is_train = tf.placeholder(tf.bool)
        is_training_dropout = tf.placeholder(tf.bool)
        mean = 120.707
        std = 64.15

        nets = (imgs-mean)/(std+1e-7)

        nets = slim.conv2d(nets,64,[3,3],activation_fn=tf.nn.relu, weights_regularizer = slim.l2_regularizer(0.001)
                           , biases_regularizer=slim.l2_regularizer(0.001))
        nets = tf.layers.batch_normalization(nets, training=is_train)
        nets = slim.dropout(nets, keep_prob= 0.3, is_training= is_training_dropout)

        nets = slim.conv2d(nets,64,[3,3],activation_fn=tf.nn.relu, weights_regularizer = slim.l2_regularizer(0.001)
                           , biases_regularizer=slim.l2_regularizer(0.001))
        nets = tf.layers.batch_normalization(nets, training=is_train)
        nets = slim.max_pool2d(nets, [2, 2])

        nets = slim.conv2d(nets,128,[3,3],activation_fn=tf.nn.relu, weights_regularizer = slim.l2_regularizer(0.001)
                           , biases_regularizer=slim.l2_regularizer(0.001))
        nets = tf.layers.batch_normalization(nets, training=is_train)
        nets = slim.dropout(nets, keep_prob= 0.4, is_training= is_training_dropout)


        nets = slim.conv2d(nets,128,[3,3],activation_fn=tf.nn.relu, weights_regularizer = slim.l2_regularizer(0.001)
                           , biases_regularizer=slim.l2_regularizer(0.001))
        nets = tf.layers.batch_normalization(nets, training=is_train)
        nets = slim.max_pool2d(nets, [2, 2])

        nets = slim.conv2d(nets,256,[3,3],activation_fn=tf.nn.relu, weights_regularizer = slim.l2_regularizer(0.001)
                           , biases_regularizer=slim.l2_regularizer(0.001))
        nets = tf.layers.batch_normalization(nets, training=is_train)
        nets = slim.dropout(nets, keep_prob= 0.4, is_training= is_training_dropout)

        nets = slim.conv2d(nets,256,[3,3],activation_fn=tf.nn.relu, weights_regularizer = slim.l2_regularizer(0.001)
                           , biases_regularizer=slim.l2_regularizer(0.001))
        nets = tf.layers.batch_normalization(nets, training=is_train)
        nets = slim.dropout(nets, keep_prob= 0.4, is_training= is_training_dropout)

        nets = slim.conv2d(nets,256,[3,3],activation_fn=tf.nn.relu, weights_regularizer = slim.l2_regularizer(0.001)
                           , biases_regularizer=slim.l2_regularizer(0.001))
        nets = tf.layers.batch_normalization(nets, training=is_train)
        nets = slim.max_pool2d(nets, [2, 2])

        nets = slim.conv2d(nets,512,[3,3],activation_fn=tf.nn.relu, weights_regularizer = slim.l2_regularizer(0.001)
                           , biases_regularizer=slim.l2_regularizer(0.001))
        nets = tf.layers.batch_normalization(nets, training=is_train)
        nets = slim.dropout(nets, keep_prob= 0.4, is_training= is_training_dropout)

        nets = slim.conv2d(nets,512,[3,3],activation_fn=tf.nn.relu, weights_regularizer = slim.l2_regularizer(0.001)
                           , biases_regularizer=slim.l2_regularizer(0.001))
        nets = tf.layers.batch_normalization(nets, training=is_train)
        nets = slim.dropout(nets, keep_prob= 0.4, is_training= is_training_dropout)

        nets = slim.conv2d(nets,512,[3,3],activation_fn=tf.nn.relu, weights_regularizer = slim.l2_regularizer(0.001)
                           , biases_regularizer=slim.l2_regularizer(0.001))
        nets = tf.layers.batch_normalization(nets, training=is_train)
        nets = slim.max_pool2d(nets, [2, 2])

        nets = slim.conv2d(nets,512,[3,3],activation_fn=tf.nn.relu, weights_regularizer = slim.l2_regularizer(0.001)
                           , biases_regularizer=slim.l2_regularizer(0.001))
        nets = tf.layers.batch_normalization(nets, training=is_train)

        nets = slim.conv2d(nets,512,[3,3],activation_fn=tf.nn.relu, weights_regularizer = slim.l2_regularizer(0.001)
                           , biases_regularizer=slim.l2_regularizer(0.001))
        nets = tf.layers.batch_normalization(nets, training=is_train)
        nets = slim.conv2d(nets,512,[3,3],activation_fn=tf.nn.relu, weights_regularizer = slim.l2_regularizer(0.001)
                           , biases_regularizer=slim.l2_regularizer(0.001))
        nets = tf.layers.batch_normalization(nets, training=is_train)
        nets = slim.max_pool2d(nets, [2, 2])
        nets = slim.dropout(nets, keep_prob= 0.5, is_training= is_training_dropout)
        nets = slim.flatten(nets)
        nets = slim.fully_connected(nets, 512, activation_fn=tf.nn.relu, weights_regularizer = slim.l2_regularizer(0.001)
                           , biases_regularizer=slim.l2_regularizer(0.001))
        nets = tf.layers.batch_normalization(nets, training=is_train)

        logits = slim.fully_connected(nets, 10)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        prob = tf.nn.softmax(logits)
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label))
        with tf.control_dependencies(update_ops):
            step = tf.train.AdamOptimizer().minimize(loss)
        return g,  prob, loss, step, is_training_dropout, is_train, imgs, label

if __name__ == '__main__':
    g,  prob, loss, step, is_training_dropout, is_training, X_vgg, Y_vgg = vgg16()

    sess = tf.Session(graph=g)

    with g.as_default():
        sess.run(tf.global_variables_initializer())

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images
            # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    bsize = 1000 #batch size
    n_batches = 50000 // bsize
    for epoch in range(200):
        s = 0
        for i in range(n_batches):
            #data, label = mnist.train.next_batch(bsize)
            z = datagen.flow(x_train[s:s+bsize],y_train[s:s+bsize],bsize)[0]
            data = z[0] #x_train[s:s+bsize]
            labels = z[1] #y_train[s:s+bsize]
            s = s+bsize
            feed_dict={X_vgg:data, Y_vgg:labels,is_training:True, is_training_dropout: True}
            sess.run(step,feed_dict)
            print('epoch %d - %d%%) '% (epoch+1, (100*(i+1))//n_batches), end='\r' if i<n_batches-1 else '')

        train_acc =(sess.run(prob,feed_dict={X_vgg: x_train[:1000],is_training:False, is_training_dropout: False}).argmax(axis = -1) == y_train[:1000].argmax(axis = -1)).sum()/1000.
        val_acc =(sess.run(prob,feed_dict={X_vgg: x_test[:1000],is_training:False, is_training_dropout: False}).argmax(axis = -1) == y_test[:1000].argmax(axis = -1)).sum()/1000.
        print('training accuracy: %2.4f \t validation accuracy: %2.4f' % (train_acc, val_acc))

    with g.as_default():
        saver = tf.train.Saver()
        saver.save(sess, 'cifar10vgg/mymodel')
