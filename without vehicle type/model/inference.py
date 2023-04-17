# -- coding: utf-8 --
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from model.temporal_attention import TemporalTransformer
class InferenceClass(object):
    def __init__(self, para=None):
        self.para=para

    def weighs_add(self, inputs, hidden_size):
        # inputs size是[batch_size, max_time, encoder_size(hidden_size)]
        u_context = tf.Variable(tf.truncated_normal([hidden_size]), name='u_context')
        # 使用一个全连接层编码GRU的输出的到期隐层表示,输出u的size是[batch_size, max_time, hidden_size]
        h = tf.layers.dense(inputs, hidden_size)
        # shape为[batch_size, max_time, 1]
        alpha = tf.nn.softmax(tf.reduce_sum(h, axis=2, keep_dims=True), dim=1)
        # alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True), dim=1)
        # reduce_sum之前shape为[batch_size, max_time, hidden_size]，之后shape为[batch_size, hidden_size]
        atten_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
        return atten_output

    def cnn(self,x=None):
        '''
        :param x: [N, output_length, site_num, emb_size]
        :return:
        '''
        with tf.variable_scope('cnn_layer', reuse=tf.AUTO_REUSE):
            filter1 = tf.Variable(initial_value=tf.random_normal(shape=[3, 108, self.para.emb_size, 64]), name='fitter_1')
            layer1 = tf.nn.conv2d(input=x, filter=filter1, strides=[1, 1, 1, 1], padding='SAME')
            layer1 = tf.nn.sigmoid(layer1)
            print('layer1 shape is : ', layer1.shape)

            # filter2 = tf.Variable(initial_value=tf.random_normal(shape=[6, 108, 64, 128]), name='fitter_2')
            # layer2 = tf.nn.conv2d(input=layer1, filter=filter2, strides=[1, 1, 1, 1], padding='SAME')
            # layer2 = tf.nn.sigmoid(layer2)
            # print('layer2 shape is : ', layer2.shape)

            # filter3 = tf.Variable(initial_value=tf.random_normal(shape=[1, 108, 256, 256]), name='fitter_3')
            # layer3 = tf.nn.conv2d(input=layer2, filter=filter3, strides=[1, 1, 1, 1], padding='VALID')
            # layer3 = tf.nn.sigmoid(layer3)
            # print('layer3 shape is : ', layer3.shape)

            layer3 = tf.reduce_mean(layer1, axis=2)

            # cnn_shape = layer3.get_shape().as_list()
            # nodes = cnn_shape[2] * cnn_shape[3]
            # cnn_out = tf.reshape(layer3, [-1, cnn_shape[1], nodes])

            results_pollution=tf.layers.dense(inputs=layer3, units=64, name='layer_pollution_1')
            results_pollution=tf.layers.dense(inputs=results_pollution, units=1, name='layer_pollution_2')
            results_pollution=tf.squeeze(results_pollution,axis=-1)
        return results_pollution

    def inference(self, out_hiddens):
        '''
        :param out_hiddens: [N, output_length, site_num, emb_size]
        :return:
        '''
        results_speed = tf.layers.dense(inputs=tf.transpose(out_hiddens, [0, 2, 1, 3]), units=64, activation=tf.nn.relu)
        results_speed = tf.layers.dense(inputs=results_speed, units=1, activation=tf.nn.relu)
        results_speed = tf.squeeze(results_speed, axis=-1, name='output_y')

        return results_speed# [N, site_num, output_length]