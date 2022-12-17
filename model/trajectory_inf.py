# -- coding: utf-8 --
import tensorflow as tf
from model.temporal_attention import TemporalTransformer



class STANClass(object):
    """
    Deep FM with FTRL optimization
    """
    def __init__(self, hp):
        """
        :param config: configuration of hyperparameters
        type of dict
        """
        # number of latent factors
        self.hp = hp
        self.emb_size = self.hp.emb_size
        self.k = self.hp.k
        self.lr = self.hp.learning_rate
        self.batch_size = self.hp.batch_size
        self.trajectory_length = self.hp.trajectory_length
        self.input_length = self.hp.input_length  # input length of speed data
        self.output_length = self.hp.output_length
        # num of features
        self.p = self.hp.feature_tra
        # num of fields
        self.field_cnt = self.hp.field_cnt

    def inference(self, X=None, feature_inds=[], keep_prob=0.0, hiddens=None):
        '''
        forward propagation
        :param X:
        :param feature_inds:
        :param keep_prob:
        :param hiddens: # (32, 12, 108, 64)
        :return: labels for each sample
        '''
        v = tf.Variable(tf.truncated_normal(shape=[self.p, self.k], mean=0, stddev=0.01), dtype='float32')

        # Factorization Machine
        if self.hp.model_name == 'DNN':
            with tf.variable_scope('DNN',reuse=False):
                # embedding layer
                y_embedding_input = tf.reshape(tf.gather(v, feature_inds), [-1, (7+self.trajectory_length*2)*self.k])
                # first hidden layer
                y_hidden_l1 = tf.layers.dense(y_embedding_input, units=self.k, activation=tf.nn.relu,
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

                # second hidden layer
                y_hidden_l2 = tf.layers.dense(y_hidden_l1, units=self.k, activation=tf.nn.relu,
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
                # third hidden layer
                y_hidden_l3 = tf.layers.dense(y_hidden_l2, units=self.k, activation=tf.nn.relu,
                                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

                # output layer
                y_fm = tf.layers.dense(y_hidden_l3, units=1, activation=tf.nn.relu,
                                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        else:
            with tf.variable_scope('FM', reuse=False):
                b = tf.get_variable('bias', shape=[1],
                                    initializer=tf.zeros_initializer())
                w1 = tf.get_variable('w1', shape=[self.p, 1],
                                     initializer=tf.truncated_normal_initializer(mean=0, stddev=1e-2))
                # shape of [None, 1]
                self.linear_terms = tf.add(tf.matmul(X, w1), b)

                # shape of [None, 1]
                self.interaction_terms = tf.multiply(0.5,
                                                     tf.reduce_mean(
                                                         tf.subtract(
                                                             tf.pow(tf.matmul(X, v), 2),
                                                             tf.matmul(tf.pow(X, 2), tf.pow(v, 2))),
                                                         1, keep_dims=True))
                # shape of [None, 1]
                y_fm = tf.add(self.linear_terms, self.interaction_terms)

        with tf.variable_scope('separate_and_sum_trajectory', reuse=False):
            self.holistic_weights =None
            x_trajectory = tf.gather(v, feature_inds)  # (N, 17, 64)
            x_common = x_trajectory[:, :-(self.trajectory_length * 2), :]  # (N, 7, 64)
            x_common = tf.reduce_sum(x_common[:, 2:, :], axis=1, keep_dims=True)  # (N, 1, 64)
            x_common = tf.concat([x_trajectory[:, :2, :], x_common], axis=1)  # (N, 3, 64)
            x_common = tf.concat(tf.split(x_common, 3, axis=1), axis=2)  # (N, 1, 3 * 64)
            x_common = tf.expand_dims(x_common, axis=1)  # (N, 1, 1, 3 * 64)   “没啥问题了”
            # x_common = tf.layers.dense(x_common, units=self.k, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(), name='x_common')
            # (N, 1, 1, 3 * 64)
            x_trajectory_distances = x_trajectory[:, -(self.trajectory_length * 2):-(self.trajectory_length * 1),
                                     :]  # (N, 5, 64)
            x_trajectory_route_id = x_trajectory[:, -(self.trajectory_length * 1):, :]  # (N, 5, 64)
            x_trajectory_separate = tf.concat([x_trajectory_distances, x_trajectory_route_id],
                                              axis=-1)  # (N, 5, 2 * 64)
            x_trajectory_separate = tf.expand_dims(x_trajectory_separate, axis=2)  # (N, 5, 1, 2 * 64)
            x_common = tf.tile(x_common, [1, self.trajectory_length, 1, 1])  # (N, 5, 1, 3 * 64)
            # x_trajectory_separate = tf.add(x=x_trajectory_separate, y=x_common)  # (N, 5, 1, 64)
            x_trajectory_separate = tf.concat([x_trajectory_separate, x_common], axis=-1)  # (N, 5, 1, 5 * 64)
            x_trajectory_separate = tf.layers.dense(x_trajectory_separate, units=self.k, activation=tf.nn.relu,
                                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            # x_trajectory_separate = tf.layers.dense(x_trajectory_separate, units=self.k, kernel_initializer=tf.truncated_normal_initializer(), name='x_common_2')
            # hiddens shape is [N, 24, 5, 64]
            hiddens = tf.transpose(hiddens, perm=[0, 2, 1, 3])
            hiddens = tf.reshape(hiddens, shape=[-1, self.output_length + self.input_length, self.emb_size])
            x_trajectory_separate = tf.reshape(x_trajectory_separate, [-1, 1, self.k])

            """using for ITTE，none of traffic speed prediction and holistic attention models"""
            # x_trajectory_separate_copy = x_trajectory_separate

            if self.hp.model_name=='Hatt':
                x_trajectory_separate = hiddens[:, self.input_length-1:self.input_length] + x_trajectory_separate
            else:
                T = TemporalTransformer(self.hp)
                x_trajectory_separate = T.encoder(hiddens=hiddens[:, -(1 + self.input_length):],
                                                  hidden=x_trajectory_separate)
                self.holistic_weights = T.attention_layer_weights
            # x_trajectory_separate = tf.squeeze(x_trajectory_separate, axis=2) # (N, 5, 64)
            x_trajectory_separate = tf.reshape(x_trajectory_separate,
                                               shape=[-1, self.trajectory_length, self.emb_size])  # (N, 5, 64)

            x_trajectory_sum = tf.layers.conv1d(inputs=x_trajectory_separate,
                                                filters=self.emb_size,
                                                kernel_size=3,
                                                padding='VALID',
                                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                name='conv_1', strides=2)

            x_trajectory_sum = tf.reduce_sum(x_trajectory_sum, axis=1)  # (N, 64)

            print('x_trajectory_sum shape is : ', x_trajectory_sum.shape)

        with tf.variable_scope('spatio_temporal_deep_fm_fusion', reuse=False):
            # separate time
            x_1 = tf.layers.dense(x_trajectory_separate, units=self.k, activation=tf.nn.relu,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            y_out_1 = tf.layers.dense(x_1, units=1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            y_out_1 = tf.squeeze(y_out_1, axis=-1)  # (N, 5)

            # sum time
            x_2 = tf.layers.dense(x_trajectory_sum, units=self.k, activation=tf.nn.relu,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            y_out_2 = tf.layers.dense(x_2, units=1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            # y_out_2 = tf.add_n([y_fm, y_out_2])

            # the combination between sptio-temporal attention network with DeepFM
            if self.hp.model_name != 'DNN':
                y_out_2 = tf.concat([self.linear_terms, self.interaction_terms, y_out_2], axis=-1)
                y_out_2 = tf.layers.dense(y_out_2, units=32, activation=tf.nn.relu,
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
                y_out_2 = tf.layers.dense(y_out_2, units=1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

            # if self.hp.model_name == 'FM':
            #     """ITTE，none of traffic speed prediction and holistic attention models"""
            #     x_trajectory_separate_copy = tf.reshape(x_trajectory_separate_copy, [-1, self.trajectory_length, self.k])
            #     trajectory_sum = tf.layers.conv1d(inputs=x_trajectory_separate_copy,
            #                                         filters=self.emb_size,
            #                                         kernel_size=3,
            #                                         padding='VALID',
            #                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            #                                         name='conv_2', strides=2)
            #     y_out_3 = tf.reduce_sum(trajectory_sum, axis=1) # (N, 64)
            #     y_out_3 = tf.layers.dense(y_out_3, units=self.k, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            #     y_out_3 = tf.layers.dense(y_out_3, units=1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            #     y_fm = tf.concat([y_out_3, y_fm], axis=-1)
            #     y_fm = tf.layers.dense(y_fm, units=32, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            #     y_fm = tf.layers.dense(y_fm, units=1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

        return y_out_1, y_out_2, y_fm