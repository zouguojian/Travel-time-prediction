# -- coding: utf-8 --
from baseline.CompactETA.model.gat import *
import numpy as np

def positional_encoding(inputs,
                        maxlen,
                        masking=True,
                        scope="positional_encoding"):
    '''Sinusoidal Positional_Encoding. See 3.5
    inputs: 3d tensor. (N, T, E)
    maxlen: scalar. Must be >= T
    masking: Boolean. If True, padding positions are set to zeros.
    scope: Optional scope for `variable_scope`.
    returns
    3d tensor that has the same shape as inputs.
    '''

    E = inputs.get_shape().as_list()[-1] # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1] # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # (N, T)

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i-i%2)/E) for i in range(E)]
            for pos in range(maxlen)])

        # Second part, apply the cosine to even columns and sin to odds.
        # position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        # position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        position_enc = np.cos(position_enc)

        position_enc = tf.convert_to_tensor(position_enc, tf.float32) # (maxlen, E)

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        # masks
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)

def adj(trajecroty_len=5, adj_matrix=None):
    '''
    :param trajecroty_len:
    :param adj_matrix: shape is [1, 5, 5, 1]
    :return:
    '''
    if adj_matrix is not None:return adj_matrix
    else:
        adj_matrix = np.zeros(shape=[trajecroty_len,trajecroty_len])
        for i in range(trajecroty_len):
            if i!=0 and i!=trajecroty_len-1:
                adj_matrix[i,i-1]=1
                adj_matrix[i,i+1]=1
                adj_matrix[i,i] =1
            elif i==0:
                adj_matrix[i,i+1]=1
                adj_matrix[i,i] =1
            else:
                adj_matrix[i,i-1]=1
                adj_matrix[i,i] =1
    adj_matrix=np.reshape(adj_matrix,[1, trajecroty_len, trajecroty_len, 1])
    return adj_matrix

class CompactETAClass(object):
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

        # GAT (we used the gat to model long road correlation which contains several local path)
        # if you have demand data like original paper data, YOU can use architecture connection relationship (0 or 1) to
        # learning the link relation embedding, but for line that do not not to use, JUST source road index map is okay, OR
        # LIKE me to transform the line to adjacent matrix.
        with tf.variable_scope('GAT', reuse=False):
            distances=tf.gather(v, feature_inds)[:,-(self.trajectory_length * 2):-self.trajectory_length]  # (N, trajectory length, 64)
            link =tf.concat([distances, speed], axis=-1) # incorporate the traffic states into global features
            link = tf.layers.dense(link, units=64, activation=tf.nn.relu,
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            Spatial = SpatialTransformer(arg=self.hp)
            # position embedding

            link=Spatial.encoder(link,adj_matrix=adj(self.trajectory_length)) # shape is [N, trajectory length, dim]

        # Position
        with tf.variable_scope('POSITION', reuse=False):
            position_emb = positional_encoding(link, maxlen=self.trajectory_length)
            link = 0.3 * tf.multiply(x=position_emb, y=link) + link
            link = tf.reduce_sum(link, axis=1)  # shape is [N, dim]

        # MLP
        with tf.variable_scope('MLP', reuse=False):
            concatnation= tf.concat([g, link],axis=-1)
            # first hidden layer
            y_hidden_l1 = tf.layers.dense(concatnation, units=64, activation=tf.nn.relu,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

            # second hidden layer
            y_hidden_l2 = tf.layers.dense(y_hidden_l1, units=64, activation=tf.nn.relu,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            # third hidden layer

            pre = tf.layers.dense(y_hidden_l2, units=1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

        # shape is [N,1]
        return pre