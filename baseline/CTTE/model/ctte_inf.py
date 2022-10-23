# -- coding: utf-8 --
from baseline.CTTE.model.gat import *
from baseline.CTTE.model.lstm import LstmClass
from baseline.CTTE.model.resent import ResnetClass

class CTTEClass(object):
    """
    Deep neural network, position, and GAT with MAPE optimization
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

    def inference(self, speed=None, feature_inds=[], keep_prob=0.0):
        '''
        forward propagation
        :param speed: speed used to represents the traffic states (N, trajectory length, dim)
        :param feature_inds:
        :param keep_prob:
        :param hiddens:
        :return: labels for each sample
        '''
        # p 总的轨迹数据元素长度
        v = tf.Variable(tf.truncated_normal(shape=[self.p, self.k], mean=0, stddev=0.01), dtype='float32')

        # global operation
        with tf.variable_scope('Global', reuse=False):
            g = tf.reshape(tf.gather(v, feature_inds[:, :-(self.trajectory_length * 2)]), [-1, 7 * self.k])
            global_h = tf.nn.relu(g)
            global_h = tf.layers.dense(global_h, units=self.emb_size, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            global_h = tf.nn.sigmoid(global_h) # shape is [N, dim]

        # LSTM
        with tf.variable_scope('LSTM', reuse=False):
            distances = tf.gather(v, feature_inds)[:, -(self.trajectory_length * 2):-self.trajectory_length] # (N, trajectory length, 64)
            links= tf.gather(v, feature_inds)[:, -self.trajectory_length:] # (N, trajectory length, 64)
            # (32, 12, 5, 64)
            speed = tf.transpose(speed, [0, 2, 1, 3])
            speed = tf.reshape(speed,shape=[self.batch_size, self.trajectory_length, self.input_length*self.emb_size])
            time_series = tf.concat([distances, links, speed], axis=-1)  # incorporate the traffic states into global features
            # [N, trajectory length, dim * n]
            # time_series=tf.gather(v, feature_inds)[:,-self.trajectory_length:,:]  # (N, trajectory length, 64)
            time_series = tf.layers.dense(time_series, units=64, activation=tf.nn.relu,
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            rnn=LstmClass(batch_size=self.batch_size, nodes=64)
            lstm_hs=rnn.encoding(inputs=time_series)

        # ATTENTION
        with tf.variable_scope('ATTENTION', reuse=False):
            Spatial = SpatialTransformer(arg=self.hp)
            spatial_hs=Spatial.encoder(lstm_hs) # shape is [N, trajectory length, dim]

        # ResNet
        with tf.variable_scope('RESNET', reuse=False):
            Resnet = ResnetClass(self.hp)
            resnet_h = Resnet.cnn(x=lstm_hs) # [N, dim]

        # MLP
        with tf.variable_scope('MLP', reuse=False):

            # speed
            pre_s = tf.layers.dense(spatial_hs, units=1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

            # travel time estimation

            global_h= tf.multiply(x=resnet_h, y=global_h)
            pre_t = tf.layers.dense(global_h, units=1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

        # shape is [N,1]
        return pre_s, pre_t