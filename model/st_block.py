# -- coding: utf-8 --
from model.spatial_attention import SpatialTransformer
import tensorflow as tf
from model.temporal_attention import TemporalTransformer
from model import tf_utils
from model.utils import *

class ST_Block():
    def __init__(self, hp=None, placeholders=None):
        '''
        :param hp:
        '''
        self.para = hp
        self.batch_size = self.para.batch_size
        self.emb_size = self.para.emb_size
        self.site_num = self.para.site_num
        self.is_training = self.para.is_training
        self.dropout = self.para.dropout
        self.hidden_size = self.para.hidden_size
        self.hidden_layer =self.para.hidden_layer
        self.feature_s = self.para.feature_s
        self.feature_tra = self.para.feature_tra
        self.placeholders = placeholders
        self.input_length = self.para.input_length

    def spatio_temporal(self, speed=None, STE=None):
        '''
        :param features: [N, site_num, emb_size]
        :return: [N, input_length, site_num, emb_size]
        '''
        # this step use to encoding the input series data
        x = tf.concat([speed, STE], axis=-1)
        # x = tf.add(speed,STE)
        # temporal correlation
        x_t = tf.transpose(speed, perm=[0, 2, 1, 3])
        x_t = tf.reshape(x_t, shape=[-1, self.input_length, self.emb_size * 1])
        T = TemporalTransformer(self.para)
        x_t = T.encoder(hiddens = x_t,
                        hidden = x_t)
        # feature fusion
        x_t = tf.layers.dense(x_t, units=self.emb_size, activation=tf.nn.relu)
        x_t = tf.layers.dense(x_t, units=self.emb_size)

        x_t = tf.reshape(x_t, shape=[self.batch_size, self.site_num, self.input_length, self.emb_size])
        x_t = tf.transpose(x_t, perm=[0, 2, 1, 3])

        """ --------------------------------------------------------------------------------------- """

        # dynamic spatial correlation
        x_s = tf.reshape(x, shape=[-1, self.site_num, self.emb_size * 2])
        S = SpatialTransformer(self.para)
        x_s = S.encoder(inputs=x_s)
        x_s = tf.layers.dense(x_s, units=self.emb_size, activation=tf.nn.relu)
        x_s = tf.layers.dense(x_s, units=self.emb_size)

        x_s = tf.reshape(x_s, shape=[-1, self.input_length, self.site_num, self.emb_size])
        # feature fusion
        x_f = gatedFusion(x_s, x_t, self.para.emb_size, False, 0.99, self.para.is_training)

        # fusion gate network
        # x = tf.multiply(x_t, x_s)
        # x = tf.sigmoid(x)
        # x_t = tf.multiply(x_t, 1-x)
        # x_s = tf.multiply(x_s, x)
        # x_f = tf.add_n([x_t, x_s])

        # x_f = tf.concat([x_t, x_s], axis=-1)
        # x_f = tf.layers.dense(x_f, units=self.emb_size, activation=tf.nn.relu)
        # x_f = tf.layers.dense(x_f, units=self.emb_size)
        # x_f = tf.reshape(x_f, shape=[self.batch_size, self.input_length, self.site_num, self.emb_size])

        return x_f #[N, input_length, site_num, emb_size]
