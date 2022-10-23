# -- coding: utf-8 --
import tensorflow as tf

class LstmClass(object):
    def __init__(self, batch_size, predict_time=1, layer_num=1, nodes=256):
        '''

        :param batch_size:
        :param layer_num:
        :param nodes:
        :param is_training:
        '''
        self.batch_size=batch_size
        self.layer_num=layer_num
        self.nodes=nodes
        self.predict_time=predict_time
        self.encoder()

    def encoder(self):
        '''
        :return:  shape is [batch size, time size, hidden size]
        '''

        def cell():
            lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=self.nodes)
            lstm_cell_=tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell,output_keep_prob=1.0)
            return lstm_cell_
        self.e_mlstm=tf.nn.rnn_cell.MultiRNNCell([cell() for _ in range(self.layer_num)])
        self.e_initial_state = self.e_mlstm.zero_state(self.batch_size, tf.float32)

    def encoding(self, inputs):
        '''
        :param inputs:
        :return: shape is [batch size, time size, hidden size]
        '''
        # out put the store data
        with tf.variable_scope('encoder_lstm'):
            self.ouputs, self.state = tf.nn.dynamic_rnn(cell=self.e_mlstm, inputs=inputs, initial_state=self.e_initial_state,dtype=tf.float32)
        return self.ouputs # shape[batch size, time, hidden size]