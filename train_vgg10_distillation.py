#!/usr/bin/env python
# coding: utf-8

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
import tensorflow_probability as tfp
import tensorflow.contrib.slim as slim
tfd = tfp.distributions



def entropy_loge(y):
    z = -np.log(y+ 1e-10)*y
    return z.sum(axis = -1)
# Build VGG16 model
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
        #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        prob = tf.nn.softmax(logits)
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label))
        #with tf.control_dependencies(update_ops):
        step = tf.train.AdamOptimizer().minimize(loss)
        return g,  prob, loss, step, is_training_dropout, is_train, imgs, label

# Build Distillation Model
def alpha_distill():
    g_distill = tf.Graph()
    with g_distill.as_default():

        imgs = tf.placeholder(tf.float32, [None, 32, 32, 3])
        label = tf.placeholder(tf.float32, [25, None, 10])
        is_train = tf.placeholder(tf.bool)
        is_training_dropout = tf.placeholder(tf.bool)
        learning_rate = tf.placeholder(tf.float32)
        mean = 120.707
        std = 64.15

        nets = (imgs-mean)/(std+1e-7)

        nets = slim.conv2d(nets,64,[3,3],activation_fn=tf.nn.relu, weights_regularizer = slim.l2_regularizer(0.001)
                           , biases_regularizer=slim.l2_regularizer(0.001))
        nets = tf.layers.batch_normalization(nets, training=is_train)
        nets = slim.dropout(nets, keep_prob= 0.5, is_training= is_training_dropout)

        nets = slim.conv2d(nets,64,[3,3],activation_fn=tf.nn.relu, weights_regularizer = slim.l2_regularizer(0.001)
                           , biases_regularizer=slim.l2_regularizer(0.001))
        nets = tf.layers.batch_normalization(nets, training=is_train)
        nets = slim.max_pool2d(nets, [2, 2])

        nets = slim.conv2d(nets,128,[3,3],activation_fn=tf.nn.relu, weights_regularizer = slim.l2_regularizer(0.001)
                           , biases_regularizer=slim.l2_regularizer(0.001))
        nets = tf.layers.batch_normalization(nets, training=is_train)
        nets = slim.dropout(nets, keep_prob= 0.5, is_training= is_training_dropout)


        nets = slim.conv2d(nets,128,[3,3],activation_fn=tf.nn.relu, weights_regularizer = slim.l2_regularizer(0.001)
                           , biases_regularizer=slim.l2_regularizer(0.001))
        nets = tf.layers.batch_normalization(nets, training=is_train)
        nets = slim.max_pool2d(nets, [2, 2])

        nets = slim.conv2d(nets,256,[3,3],activation_fn=tf.nn.relu, weights_regularizer = slim.l2_regularizer(0.001)
                           , biases_regularizer=slim.l2_regularizer(0.001))
        nets = tf.layers.batch_normalization(nets, training=is_train)
        nets = slim.dropout(nets, keep_prob= 0.5, is_training= is_training_dropout)

        nets = slim.conv2d(nets,256,[3,3],activation_fn=tf.nn.relu, weights_regularizer = slim.l2_regularizer(0.001)
                           , biases_regularizer=slim.l2_regularizer(0.001))
        nets = tf.layers.batch_normalization(nets, training=is_train)
        nets = slim.dropout(nets, keep_prob= 0.5, is_training= is_training_dropout)

        nets = slim.conv2d(nets,256,[3,3],activation_fn=tf.nn.relu, weights_regularizer = slim.l2_regularizer(0.001)
                           , biases_regularizer=slim.l2_regularizer(0.001))
        nets = tf.layers.batch_normalization(nets, training=is_train)
        nets = slim.max_pool2d(nets, [2, 2])

        nets = slim.conv2d(nets,512,[3,3],activation_fn=tf.nn.relu, weights_regularizer = slim.l2_regularizer(0.001)
                           , biases_regularizer=slim.l2_regularizer(0.001))
        nets = tf.layers.batch_normalization(nets, training=is_train)
        nets = slim.dropout(nets, keep_prob= 0.5, is_training= is_training_dropout)

        nets = slim.conv2d(nets,512,[3,3],activation_fn=tf.nn.relu, weights_regularizer = slim.l2_regularizer(0.001)
                           , biases_regularizer=slim.l2_regularizer(0.001))
        nets = tf.layers.batch_normalization(nets, training=is_train)
        nets = slim.dropout(nets, keep_prob= 0.5, is_training= is_training_dropout)

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


        alpha = slim.fully_connected(nets, 10, activation_fn=tf.nn.softplus)

        S = tf.reduce_sum(alpha, axis = -1, keep_dims= True)


        distribution = tfd.Dirichlet(alpha)
        loss = -tf.reduce_mean(distribution.log_prob(label[0,:,:]))
        for i in range(24):
            loss += -tf.reduce_mean(distribution.log_prob(label[i,:,:]))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        return g_distill,  alpha, loss, step, is_training_dropout, is_train, imgs, label, learning_rate

if __name__ == '__main__':
    # Create graph
    g,  prob, loss, step, is_training_dropout, is_training, X_vgg, Y_vgg = vgg16()
    sess = tf.Session(graph=g)
    with g.as_default():
        saver = tf.train.Saver()
        saver.restore(sess, 'cifar10vgg/mymodel')

    g_distill,  mean_prob_distill, loss_distill, step_distill,is_training_dropout_distill, is_training_distill, X_distill, Y_distill, learning_rate =  alpha_distill()
    sess_distill = tf.Session(graph=g_distill)
    with g_distill.as_default():
        sess_distill.run(tf.global_variables_initializer())
        saver_distill = tf.train.Saver()

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

    #start training
    bsize = 1000 #batch size
    n_batches = 50000 // bsize
    for iteration in range(250):
        s = 0
        for i in range(n_batches):
            z = datagen.flow(x_train[s:s+bsize],y_train[s:s+bsize],bsize)[0]
            clean_img = z[0]
            labels = z[1]
            s = s+bsize
            data = clean_img
            y_prob = []
            for j in range(25):
                y_prob.append(sess.run(prob,feed_dict={X_vgg: data,is_training:False, is_training_dropout: True}))
            y_prob = np.array(y_prob)
            feed_dict={X_distill:data, Y_distill:y_prob,is_training_distill:True, is_training_dropout_distill: True, learning_rate: 0.001}
            sess_distill.run([step_distill],feed_dict)
            print('iteration %d - %d%%) '% (iteration+1, (100*(i+1))//n_batches), end='\r' if i<n_batches-1 else '')
        if (iteration+1) %20 == 0:
            saver_distill.save(sess_distill, 'cifar10vgg_distill/mymodel_distill')
        train_acc =(sess_distill.run(mean_prob_distill,feed_dict={X_distill: x_train[:bsize],is_training_distill:False, is_training_dropout_distill: False}).argmax(axis = -1) == y_train[:bsize].argmax(axis = -1)).sum()/1000.
        val_acc =(sess_distill.run(mean_prob_distill,feed_dict={X_distill: x_test[:bsize],is_training_distill:False, is_training_dropout_distill: False}).argmax(axis = -1) == y_test[:bsize].argmax(axis = -1)).sum()/1000.
        print('training accuracy: %2.4f \t validation accuracy: %2.4f' % (train_acc, val_acc))
    with g_distill.as_default():
        saver_distill = tf.train.Saver()
        saver_distill.save(sess_distill, 'cifar10vgg_distill/mymodel_distill')

    #Plot the distribution

    #Distillation predictions
    alpha_cifar100 = sess_distill.run(mean_prob_distill,feed_dict={X_distill: x_test_cifar100[2000:3000],is_training_distill:False, is_training_dropout_distill: False})
    alpha_cifar10 = sess_distill.run(mean_prob_distill,feed_dict={X_distill: x_test[2000:3000],is_training_distill:False, is_training_dropout_distill: False})


    #MC-Dropout predictions
    z_cifar100 = []
    for i in range(50):
        z_cifar100.append(sess.run(prob,feed_dict={X_vgg: x_test_cifar100[2000:3000],is_training:True, is_training_dropout: True}))

    z_cifar10 = []
    for i in range(50):
        z_cifar10.append(sess.run(prob,feed_dict={X_vgg: x_test[2000:3000],is_training:True, is_training_dropout: True}))

    z_cifar100 = np.array(z_cifar100)
    z_cifar10 = np.array(z_cifar10)

    plt.figure(figsize=(20,12))
    plt.subplot(2,3,1)
    plt.hist((prob_cifar10*(1-prob_cifar10)/(alpha_cifar10.sum(axis = -1, keepdims= True)+1.)).mean(axis = -1)
             ,bins = np.arange(10)*0.0045,density = True,color='red', alpha = 0.2, label='Distillation' )
    plt.hist(z_cifar10.var(axis = 0).mean(axis = -1),bins = np.arange(10)*0.0045
             , density = True,color= 'b', alpha = 0.2, label='MC Dropout')
    plt.legend()
    plt.xlabel('Mutual Information')
    plt.ylabel('In-distribution - CIFAR 10')

    plt.subplot(2,3,2)
    plt.hist(entropy_loge(prob_cifar10) - (prob_cifar10*(1-prob_cifar10)/(alpha_cifar10.sum(axis = -1, keepdims= True)+1.)).mean(axis = -1)
             ,bins = np.arange(10)*0.1,density = True,color='red', alpha = 0.2, label='Distillation' )
    plt.hist(entropy_loge(z_cifar10.mean(axis = 0)) - z_cifar10.var(axis = 0).mean(axis = -1)
             ,bins = np.arange(10)*0.1, density = True,color= 'b', alpha = 0.2, label='MC Dropout')
    plt.legend()
    plt.xlabel('Aleatoric Entropy')
    #plt.title('Out-of-distribution Mutual Information - CIFAR 100')

    plt.subplot(2,3,3)
    plt.hist(entropy_loge(prob_cifar10),bins =np.arange(10)*0.1
             ,density = True,color='red', alpha = 0.2, label='Distillation' )
    plt.hist(entropy_loge(z_cifar10.mean(axis = 0)),bins = np.arange(10)*0.1
             , density = True,color= 'b', alpha = 0.2, label='MC Dropout')
    plt.legend()
    plt.xlabel('Predictive Entropy')


    plt.subplot(2,3,4)
    plt.hist((prob_cifar100*(1-prob_cifar100)/(alpha_cifar100.sum(axis = -1, keepdims= True)+1.)).mean(axis = -1)
             ,bins = np.arange(10)*0.0045,density = True,color='red', alpha = 0.2, label='Distillation' )
    plt.hist(z_cifar100.var(axis = 0).mean(axis = -1),bins = np.arange(10)*0.0045
             , density = True,color= 'b', alpha = 0.2, label='MC Dropout')
    plt.legend()
    plt.xlabel('Mutual Information')
    plt.ylabel('Out-of-distribution - CIFAR 100')
    plt.ylim([0,175])

    plt.subplot(2,3,5)
    plt.hist(entropy_loge(prob_cifar100) - (prob_cifar100*(1-prob_cifar100)/(alpha_cifar100.sum(axis = -1, keepdims= True)+1.)).sum(axis = -1)
             ,bins = np.arange(10)*0.1,density = True,color='red', alpha = 0.2, label='Distillation' )
    plt.hist(entropy_loge(z_cifar100.mean(axis = 0)) - z_cifar100.var(axis = 0).mean(axis = -1)
             ,bins = np.arange(10)*0.1, density = True,color= 'b', alpha = 0.2, label='MC Dropout')
    plt.legend()
    plt.xlabel('Aleatoric Entropy')
    plt.ylim([0,5.5])

    plt.subplot(2,3,6)
    plt.hist(entropy_loge(prob_cifar100),bins = np.arange(10)*0.1,density = True,color='red', alpha = 0.2, label='Distillation' )
    plt.hist(entropy_loge(z_cifar100.mean(axis = 0)),bins = np.arange(10)*0.1, density = True,color= 'b', alpha = 0.2, label='MC Dropout')
    plt.legend()
    plt.xlabel('Predictive Entropy')
    plt.ylim([0,7.5])
