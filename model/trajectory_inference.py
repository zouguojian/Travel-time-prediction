# -- coding: utf-8 --
import tensorflow as tf
from model.temporal_attention import TemporalTransformer

class DeepFM(object):
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
        self.input_length = self.hp.input_length             # input length of speed data
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
        v = tf.Variable(tf.truncated_normal(shape=[self.p, self.k], mean=0, stddev=0.01),dtype='float32')
                                                   # p 总的轨迹数据元素长度
        # Factorization Machine
        with tf.variable_scope('FM', reuse=False):
            b = tf.get_variable('bias', shape=[1],
                                initializer=tf.zeros_initializer())
            w1 = tf.get_variable('w1', shape=[self.p, 1],
                                 initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
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

        # three-hidden-layer neural network, network shape of (200-200-200)
        with tf.variable_scope('DNN',reuse=False):
            # embedding layer, 切分出对应字段的的隐藏空间特征值
            y_embedding_input = tf.reshape(tf.gather(v, feature_inds), [-1, self.field_cnt*self.k])
            # first hidden layer
            w1 = tf.get_variable('w1_dnn', shape=[self.field_cnt*self.k, self.emb_size],
                                 initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
            b1 = tf.get_variable('b1_dnn', shape=[self.emb_size],
                                 initializer=tf.constant_initializer(0.001))
            y_hidden_l1 = tf.nn.relu(tf.matmul(y_embedding_input, w1) + b1)
            # second hidden layer
            w2 = tf.get_variable('w2', shape=[self.emb_size, self.emb_size],
                                 initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
            b2 = tf.get_variable('b2', shape=[self.emb_size],
                                 initializer=tf.constant_initializer(0.001))
            y_hidden_l2 = tf.nn.relu(tf.matmul(y_hidden_l1, w2) + b2)
            # third hidden layer
            w3 = tf.get_variable('w1', shape=[self.emb_size, self.emb_size],
                                 initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
            b3 = tf.get_variable('b1', shape=[self.emb_size],
                                 initializer=tf.constant_initializer(0.001))
            y_hidden_l3 = tf.nn.relu(tf.matmul(y_hidden_l2, w3) + b3)
            # output layer
            w_out = tf.get_variable('w_out', shape=[self.emb_size, 1],
                                 initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
            b_out = tf.get_variable('b_out', shape=[1],
                                 initializer=tf.constant_initializer(0.001))
            y_dnn = tf.nn.relu(tf.matmul(y_hidden_l3, w_out) + b_out)

        with tf.variable_scope('separate_and_sum_trajectory',reuse=False):
            x_trajectory = tf.gather(v, feature_inds)  # (N, 17, 64)
            x_common = x_trajectory[:, :-(self.trajectory_length * 2), :] # (N, 10, 64)
            x_common = tf.reshape(x_common, [-1, (self.field_cnt - self.trajectory_length * 2) * self.k]) # (N, 10 * 64)
            x_common = tf.expand_dims(x_common, axis=1) # (N, 1, 10 * 64)
            x_common = tf.expand_dims(x_common, axis=1)  # (N, 1, 1, 10 * 64)

            x_common = tf.layers.dense(x_common, units=self.k, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), name='x_common')
            x_trajectory_distances = x_trajectory[:, -(self.trajectory_length * 2):-(self.trajectory_length * 1), :] # (N, 5, 64)
            x_trajectory_route_id = x_trajectory[:, -(self.trajectory_length * 1):, :] # (N, 5, 64)
            x_trajectory_separate = tf.add(x=x_trajectory_distances, y=x_trajectory_route_id) # (N, 5, 64)
            x_trajectory_separate = tf.expand_dims(x_trajectory_separate, axis=2)  # (N, 5, 1, 64)
            x_trajectory_separate = tf.add(x=x_trajectory_separate, y=x_common)  # (N, 5, 1, 64)

            hiddens = tf.transpose(hiddens, perm=[0, 2, 1, 3])
            hiddens = tf.reshape(hiddens, shape=[-1, self.input_length + self.output_length, self.emb_size])
            x_trajectory_separate = tf.reshape(x_trajectory_separate, [-1, 1, self.k])
            T = TemporalTransformer(self.hp)
            x_trajectory_separate = T.encoder(hiddens=hiddens, hidden=x_trajectory_separate)
            x_trajectory_separate = tf.reshape(x_trajectory_separate, shape=[-1, self.trajectory_length, self.emb_size]) # (N, 5, 64)

            x_trajectory_separate = tf.layers.conv1d(inputs=x_trajectory_separate,
                                                        filters=self.emb_size,
                                                        kernel_size=3,
                                                        padding='VALID',
                                                        kernel_initializer=tf.truncated_normal_initializer(),
                                                        name='conv_1',strides=2)

            x_trajectory_sum = tf.reduce_sum(x_trajectory_separate, axis=1) # (N, 64)

            print('x_trajectory_sum shape is : ', x_trajectory_sum.shape)

        with tf.variable_scope('spatio_temporal_deep_fm_fusion', reuse=False):
            # add FM output and DNN output
            y_out = tf.add_n([y_fm, y_dnn])+x_trajectory_sum
            y_out = tf.layers.dense(y_out, units=1, kernel_initializer=tf.truncated_normal_initializer(), name='y_out')

        print(y_out.shape)
        return y_out