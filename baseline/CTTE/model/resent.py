# -- coding: utf-8 --
import tensorflow as tf


class ResnetClass(object):
    def __init__(self, hp=None):
        '''
        :param batch_size:
        :param para:
        '''
        self.hp =hp
        self.emb_size = self.hp.emb_size
        self.trajectory_length =self.hp.trajectory_length
        self.h = [1, 3, 1]
        self.w = [1, 3, 1]

    def block(self, x, channels=[32, 32, 64], block_name=''):
        '''
        :param x:
        :param channel:
        :param block_name:
        :return:
        '''
        x1 = tf.layers.conv1d(inputs=x,
                            filters=channels[0],
                            kernel_size=1,
                            padding='SAME',
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            name='conv_1'+block_name,
                            strides=1,activation=tf.nn.relu)

        x2 = tf.layers.conv1d(inputs=x1,
                            filters=channels[1],
                            kernel_size=3,
                            padding='SAME',
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            name='conv_2'+block_name,
                            strides=1,activation=tf.nn.relu)

        x3 = tf.layers.conv1d(inputs=x2,
                            filters=channels[2],
                            kernel_size=1,
                            padding='SAME',
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            name='conv_3'+block_name,
                            strides=1,activation=tf.nn.relu)
        return x3

    def residual_connected(self, x1, x2, channel, residual_name):
        '''
        :param x1:
        :param x2:
        :param channel:
        :param residual_name:
        :return:
        '''
        x = tf.layers.conv1d(inputs=x1,
                            filters=channel,
                            kernel_size=1,
                            padding='SAME',
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            name='conv'+residual_name,
                            strides=1,activation=tf.nn.relu)
        x = x + x2
        return x

    def cnn(self, x):
        '''
        :param x: [batch size, trajectory, dim]
        :return: [batch size, dim]
        '''
        with tf.variable_scope(name_or_scope='resnet', reuse=False):
            block1 = self.block(x, channels=[32, 32, 64], block_name='block1')
            residual1 = self.residual_connected(x, block1, channel=64, residual_name='residual1')
            # print('residual 1 shape is : ', residual1.shape)

            block2 = self.block(residual1, channels=[32, 32, 64], block_name='block2')
            residual2 = self.residual_connected(residual1, block2, channel=64,residual_name='residual2')
            # print('residual 2 shape is : ', residual2.shape)

            block3 = self.block(residual2, channels=[32, 32, self.emb_size], block_name='block3')
            residual3 = self.residual_connected(residual2, block3, channel=self.emb_size, residual_name='residual3')
            # print('residual 3 shape is : ', residual3.shape)

            avr_pool = tf.layers.average_pooling1d(inputs=residual3,
                                                   pool_size=self.trajectory_length,
                                                   strides=1,
                                                   padding='valid',
                                                   name='pooling')
            avr_pool = tf.squeeze(avr_pool, axis=1)
            # print('max_pool output shape is : ', avr_pool.shape)
        return avr_pool