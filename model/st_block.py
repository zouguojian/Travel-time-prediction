# -- coding: utf-8 --
from models.spatial_attention import SpatialTransformer
import tensorflow as tf
from models.temporal_attention import TemporalTransformer
from models.lstm import LstmClass
from baseline.gman import tf_utils
from models.utils import *

class ST_Block():
    def __init__(self, hp=None, placeholders=None, input_length=6, model_func=None):
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
        self.features = self.para.features
        self.features_p = self.para.features_p
        self.placeholders = placeholders
        self.input_length = input_length
        self.model_func = model_func

    def FC(self, x, units, activations, bn, bn_decay, is_training, use_bias=True):
        if isinstance(units, int):
            units = [units]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            activations = list(activations)
        assert type(units) == list
        for num_unit, activation in zip(units, activations):
            x = tf_utils.conv2d(
                x, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
                padding='VALID', use_bias=use_bias, activation=activation,
                bn=bn, bn_decay=bn_decay, is_training=is_training)
        return x

    def gatedFusion(self, HS, HT, D, bn, bn_decay, is_training):
        '''
        gated fusion
        HS:     [batch_size, num_step, N, D]
        HT:     [batch_size, num_step, N, D]
        D:      output dims
        return: [batch_size, num_step, N, D]
        '''
        XS = self.FC(
            HS, units=D, activations=None,
            bn=bn, bn_decay=bn_decay,
            is_training=is_training, use_bias=False)
        XT = self.FC(
            HT, units=D, activations=None,
            bn=bn, bn_decay=bn_decay,
            is_training=is_training, use_bias=True)
        z = tf.nn.sigmoid(tf.add(XS, XT))
        H = tf.add(tf.multiply(z, HS), tf.multiply(1 - z, HT))
        H = self.FC(
            H, units=[D, D], activations=[tf.nn.relu, None],
            bn=bn, bn_decay=bn_decay, is_training=is_training)
        return H

    def STEmbedding(self, SE, TE, T, D, bn, bn_decay, is_training):
        '''
        spatio-temporal embedding
        SE:     [N, D]
        TE:     [batch_size, P + Q, 2] (dayofweek, timeofday)
        T:      num of time steps in one day
        D:      output dims
        retrun: [batch_size, P + Q, N, D]
        '''
        # spatial embedding
        SE = self.FC(
            SE, units=[D, D], activations=[tf.nn.relu, None],
            bn=bn, bn_decay=bn_decay, is_training=is_training)
        # temporal embedding
        TE = tf.add_n(TE)
        # TE = tf.concat((TE), axis=-1)
        TE = self.FC(
            TE, units=[D, D], activations=[tf.nn.relu, None],
            bn=bn, bn_decay=bn_decay, is_training=is_training)
        return tf.add(SE, TE)

    def spatio_temporal_(self, speed=None, STE=None, supports=None):
        X = speed
        for _ in range(self.para.num_blocks):
            X = STAttBlock(X, STE, self.para.num_heads, self.para.emb_size // self.para.num_heads, False, 0.99, self.para.is_training)
        return X

    def spatio_temporal(self, speed=None, STE=None, supports=None):
        '''
        :param features: [N, site_num, emb_size]
        :param day: [N, input_length, site_num, emb_size]
        :param hour:
        :param position:
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
        # time series correlation
        x_l = tf.transpose(x, perm=[0, 2, 1, 3])
        x_l = tf.reshape(x_l, shape=[-1, self.input_length, self.emb_size * 2])
        # x_l = tf.reshape(tf.transpose(speed,[0,2,1,3]), shape=[-1, self.para.input_length, self.para.emb_size])
        lstm_init = LstmClass(batch_size=self.batch_size * self.site_num,
                              layer_num=self.hidden_layer,
                              nodes=self.hidden_size,
                              placeholders=self.placeholders)
        x_l, c_states = lstm_init.encoding(x_l)
        # feature fusion
        x_t = tf.add_n([x_t, x_l])
        # x_t = tf.concat([x_t, x_l], axis=-1)
        x_t = tf.layers.dense(x_t, units=self.emb_size, activation=tf.nn.relu)
        x_t = tf.layers.dense(x_t, units=self.emb_size)
        x_t = tf.reshape(x_t, shape=[self.batch_size, self.site_num, self.input_length, self.emb_size])
        x_t = tf.transpose(x_t, perm=[0, 2, 1, 3])

        """ --------------------------------------------------------------------------------------- """

        # dynamic spatial correlation
        x_s = tf.reshape(x, shape=[-1, self.site_num, self.emb_size * 2])
        S = SpatialTransformer(self.para)
        x_s = S.encoder(inputs=x_s)
        # static spatial correlation
        x_g = tf.reshape(speed, shape=[-1, self.site_num, self.emb_size * 1])
        gcn = self.model_func(self.placeholders,
                              input_dim=self.emb_size * 1,
                              para=self.para,
                              supports=supports)
        x_g = gcn.predict(x_g)


        # x_g = tf.concat(tf.split(x_g, self.para.num_heads, axis=2), axis=0)
        # encoder_gcn = self.model_func(placeholders=self.placeholders,
        #                                 input_dim=self.emb_size // self.para.num_heads,
        #                                 para=self.para,
        #                                 supports=supports)
        # x_g = encoder_gcn.predict(x_g)
        # x_g = tf.concat(tf.split(x_g, self.para.num_heads, axis=0), axis=2)

        # x_s = tf.add_n([x_s, x_g])
        x_s = tf.concat([x_s, x_g], axis=-1)
        x_s = tf.layers.dense(x_s, units=self.emb_size, activation=tf.nn.relu)
        x_s = tf.layers.dense(x_s, units=self.emb_size)

        x_s = tf.reshape(x_s, shape=[-1, self.input_length, self.site_num, self.emb_size])
        # feature fusion
        x_f = self.gatedFusion(x_s, x_t, self.para.emb_size, False, 0.99, self.para.is_training)
        # x_f = tf.concat([x_t, x_s], axis=-1)
        # x_f = tf.layers.dense(x_f, units=self.emb_size, activation=tf.nn.relu)
        # x_f = tf.layers.dense(x_f, units=self.emb_size)
        # x_f = tf.reshape(x_f, shape=[self.batch_size, self.input_length, self.site_num, self.emb_size])

        return x_f #[N, input_length, site_num, emb_size]

def spatialAttention(X, STE, K, d, bn, bn_decay, is_training):
    '''
    spatial attention mechanism
    X:      [batch_size, num_step, N, D]
    STE:    [batch_size, num_step, N, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, N, D]
    '''
    D = K * d
    X = tf.concat((X, STE), axis=-1) # 和我们的一致
    # [batch_size, num_step, N, K * d]
    query = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    key = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    value = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    # [K * batch_size, num_step, N, d]
    query = tf.concat(tf.split(query, K, axis=-1), axis=0)
    key = tf.concat(tf.split(key, K, axis=-1), axis=0)
    value = tf.concat(tf.split(value, K, axis=-1), axis=0)
    # [K * batch_size, num_step, N, N]
    attention = tf.matmul(query, key, transpose_b=True)
    attention /= (d ** 0.5)
    attention = tf.nn.softmax(attention, axis=-1)
    # [batch_size, num_step, N, D]
    X = tf.matmul(attention, value)
    X = tf.concat(tf.split(X, K, axis=0), axis=-1)
    X = FC(
        X, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return X


def temporalAttention(X, STE, K, d, bn, bn_decay, is_training, mask=True):
    '''
    temporal attention mechanism
    X:      [batch_size, num_step, N, D]
    STE:    [batch_size, num_step, N, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, N, D]
    '''
    D = K * d
    X = tf.concat((X, STE), axis=-1) # 和我们的一致
    # [batch_size, num_step, N, K * d]
    query = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    key = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    value = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    # [K * batch_size, num_step, N, d]
    query = tf.concat(tf.split(query, K, axis=-1), axis=0)
    key = tf.concat(tf.split(key, K, axis=-1), axis=0)
    value = tf.concat(tf.split(value, K, axis=-1), axis=0)
    # query: [K * batch_size, N, num_step, d]
    # key:   [K * batch_size, N, d, num_step]
    # value: [K * batch_size, N, num_step, d]
    query = tf.transpose(query, perm=(0, 2, 1, 3))
    key = tf.transpose(key, perm=(0, 2, 3, 1))
    value = tf.transpose(value, perm=(0, 2, 1, 3))
    # [K * batch_size, N, num_step, num_step]
    attention = tf.matmul(query, key)
    attention /= (d ** 0.5)
    # mask attention score
    if mask:
        batch_size = tf.shape(X)[0]
        num_step = X.get_shape()[1].value
        N = X.get_shape()[2].value
        mask = tf.ones(shape=(num_step, num_step))
        mask = tf.linalg.LinearOperatorLowerTriangular(mask).to_dense()
        mask = tf.expand_dims(tf.expand_dims(mask, axis=0), axis=0)
        mask = tf.tile(mask, multiples=(K * batch_size, N, 1, 1))
        mask = tf.cast(mask, dtype=tf.bool)
        attention = tf.where(
            condition=mask, x=attention, y=-2 ** 15 + 1)
    # softmax
    attention = tf.nn.softmax(attention, axis=-1)
    # [batch_size, num_step, N, D]
    X = tf.matmul(attention, value)
    X = tf.transpose(X, perm=(0, 2, 1, 3))
    X = tf.concat(tf.split(X, K, axis=0), axis=-1)
    X = FC(
        X, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return X

def gatedFusion(HS, HT, D, bn, bn_decay, is_training):
    '''
    gated fusion
    HS:     [batch_size, num_step, N, D]
    HT:     [batch_size, num_step, N, D]
    D:      output dims
    return: [batch_size, num_step, N, D]
    '''
    XS = FC(
        HS, units=D, activations=None,
        bn=bn, bn_decay=bn_decay,
        is_training=is_training, use_bias=False)
    XT = FC(
        HT, units=D, activations=None,
        bn=bn, bn_decay=bn_decay,
        is_training=is_training, use_bias=True)
    z = tf.nn.sigmoid(tf.add(XS, XT))
    H = tf.add(tf.multiply(z, HS), tf.multiply(1 - z, HT))
    H = FC(
        H, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return H

def STAttBlock(X, STE, K, d, bn, bn_decay, is_training, mask=False):
    HS = spatialAttention(X, STE, K, d, bn, bn_decay, is_training)
    HT = temporalAttention(X, STE, K, d, bn, bn_decay, is_training, mask=mask)
    H = gatedFusion(HS, HT, K * d, bn, bn_decay, is_training)
    return tf.add(X, H)