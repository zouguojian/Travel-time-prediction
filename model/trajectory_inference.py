# -- coding: utf-8 --
import tensorflow as tf

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
        self.k = self.hp.k
        self.lr = self.hp.learning_rate
        self.batch_size = self.hp.batch_size
        # num of features
        self.p = self.hp.feature_tra
        # num of fields
        self.field_cnt = self.hp.field_cnt

    # def add_placeholders(self):
    #     self.X = tf.placeholder('float32', [None, self.p])
    #     self.y = tf.placeholder('int64', [None,])
    #     # index of none-zero features
    #     self.feature_inds = tf.placeholder('int64', [None, self.field_cnt])
    #     self.keep_prob = tf.placeholder('float32')

    def inference(self, X=None, feature_inds=[], keep_prob=0.0):
        '''
        forward propagation
        :param X:
        :param feature_inds:
        :param keep_prob:
        :return: labels for each sample
        '''
        v = tf.Variable(tf.truncated_normal(shape=[self.p, self.k], mean=0, stddev=0.01),dtype='float32')
                                                   # p 总的轨迹数据元素长度
        # Factorization Machine
        with tf.variable_scope('FM'):
            b = tf.get_variable('bias', shape=[2],
                                initializer=tf.zeros_initializer())
            w1 = tf.get_variable('w1', shape=[self.p, 2],
                                 initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
            # shape of [None, 2]
            self.linear_terms = tf.add(tf.matmul(X, w1), b)

            # shape of [None, 1]
            self.interaction_terms = tf.multiply(0.5,
                                                 tf.reduce_mean(
                                                     tf.subtract(
                                                         tf.pow(tf.matmul(X, v), 2),
                                                         tf.matmul(tf.pow(X, 2), tf.pow(v, 2))),
                                                     1, keep_dims=True))
            # shape of [None, 2]
            self.y_fm = tf.add(self.linear_terms, self.interaction_terms)

        # three-hidden-layer neural network, network shape of (200-200-200)
        with tf.variable_scope('DNN',reuse=False):
            # embedding layer, 切分出对应字段的的隐藏空间特征值
            y_embedding_input = tf.reshape(tf.gather(v, feature_inds), [-1, self.field_cnt*self.k])
            # first hidden layer
            w1 = tf.get_variable('w1_dnn', shape=[self.field_cnt*self.k, 200],
                                 initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
            b1 = tf.get_variable('b1_dnn', shape=[200],
                                 initializer=tf.constant_initializer(0.001))
            y_hidden_l1 = tf.nn.relu(tf.matmul(y_embedding_input, w1) + b1)
            # second hidden layer
            w2 = tf.get_variable('w2', shape=[200, 200],
                                 initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
            b2 = tf.get_variable('b2', shape=[200],
                                 initializer=tf.constant_initializer(0.001))
            y_hidden_l2 = tf.nn.relu(tf.matmul(y_hidden_l1, w2) + b2)
            # third hidden layer
            w3 = tf.get_variable('w1', shape=[200, 200],
                                 initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
            b3 = tf.get_variable('b1', shape=[200],
                                 initializer=tf.constant_initializer(0.001))
            y_hidden_l3 = tf.nn.relu(tf.matmul(y_hidden_l2, w3) + b3)
            # output layer
            w_out = tf.get_variable('w_out', shape=[200, 2],
                                 initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
            b_out = tf.get_variable('b_out', shape=[2],
                                 initializer=tf.constant_initializer(0.001))
            self.y_dnn = tf.nn.relu(tf.matmul(y_hidden_l3, w_out) + b_out)
        # add FM output and DNN output
        self.y_out = tf.add(self.y_fm, self.y_dnn)
        self.y_out_prob = tf.nn.softmax(self.y_out)