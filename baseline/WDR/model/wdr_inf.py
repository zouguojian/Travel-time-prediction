# -- coding: utf-8 --
from baseline.WDR.model.lstm import *
class WDRClass(object):
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

    def inference(self, X=None, feature_inds=[], keep_prob=0.0, speed=None):
        '''
        forward propagation
        :param X:
        :param feature_inds:
        :param keep_prob:
        :param hiddens: # (32, 12, 108, 64)
        :return: labels for each sample
        '''
        v = tf.Variable(tf.truncated_normal(shape=[self.p, self.k], mean=0, stddev=0.01), dtype='float32')
        # p 总的轨迹数据元素长度

        # Factorization Machine (FM)
        with tf.variable_scope('Wide', reuse=False):
            # self.linear_terms = tf.layers.dense(X, units=256, activation=tf.nn.relu,
            #                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            #                                     bias_initializer=tf.zeros_initializer())

            b = tf.get_variable('bias', shape=[256],
                                initializer=tf.zeros_initializer())
            w1 = tf.get_variable('w1', shape=[self.p, 256],
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
            # shape of [None, 256]
            y_fm = tf.add(self.linear_terms, self.interaction_terms)

        # Wide & Deep Learning
        with tf.variable_scope('Deep',reuse=False):
            # embedding layer
            y_embedding_input = tf.reshape(tf.gather(v, feature_inds), [-1, (7+self.trajectory_length*2)*self.k])
            # first hidden layer
            y_hidden_l1 = tf.layers.dense(y_embedding_input, units=256, activation=tf.nn.relu,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

            # second hidden layer
            y_hidden_l2 = tf.layers.dense(y_hidden_l1, units=256, activation=tf.nn.relu,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            # third hidden layer
            y_hidden_l3 = tf.layers.dense(y_hidden_l2, units=256, activation=tf.nn.relu,
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

        # lstm
        with tf.variable_scope('LSTM', reuse=False):
            distances = tf.gather(v, feature_inds)[:, -(self.trajectory_length * 2):-self.trajectory_length] # (N, trajectory length, 64)
            time_series = tf.concat([distances, speed], axis=-1)  # incorporate the traffic states into global features
            # time_series=tf.gather(v, feature_inds)[:,-self.trajectory_length:,:]  # (N, trajectory length, 64)
            time_series = tf.layers.dense(time_series, units=256, activation=tf.nn.relu,
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            rnn=LstmClass(batch_size=self.batch_size, nodes=256)
            lstm_h=rnn.encoding(inputs=time_series)

        # MLP
        with tf.variable_scope('MLP', reuse=False):
            concatnation= tf.concat([y_fm,y_hidden_l3,lstm_h],axis=-1)
            pre = tf.layers.dense(concatnation, units=1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

        # shape is [N,1]
        return pre