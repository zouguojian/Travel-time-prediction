# -- coding: utf-8 --
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import argparse

from model.embedding import embedding
from model.deepfm import DeepFM
from model.hyparameter import parameter

tf.reset_default_graph()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Model(object):
    def __init__(self, hp):
        self.hp = hp
        self.site_num = self.hp.site_num
        self.input_length = self.hp.input_length
        self.output_length = self.hp.output_length

        # define placeholders
        self.placeholders = {
            'position': tf.placeholder(tf.int32, shape=(1, self.site_num), name='input_position'),
            'day': tf.placeholder(tf.int32, shape=(None, self.site_num), name='input_day'),
            'hour': tf.placeholder(tf.int32, shape=(None, self.site_num), name='input_hour'),
            'minute': tf.placeholder(tf.int32, shape=(None, self.site_num), name='input_minute'),
            'indices_i': tf.placeholder(dtype=tf.int64, shape=[None, None], name='input_indices'),
            'values_i': tf.placeholder(dtype=tf.float32, shape=[None], name='input_values'),
            'dense_shape_i': tf.placeholder(dtype=tf.int64, shape=[None], name='input_dense_shape'),
            'features_s': tf.placeholder(tf.float32,
                                         shape=[None, self.input_length, self.site_num, self.para.features],
                                         name='input_s'),
            'labels_s': tf.placeholder(tf.float32, shape=[None, self.site_num, self.output_length],
                                       name='labels_s'),
            'features_p': tf.placeholder(tf.float32, shape=[None, self.input_length, self.para.features_p],
                                         name='input_p'),
            'labels_p': tf.placeholder(tf.float32, shape=[None, self.output_length], name='labels_p'),
            'dropout': tf.placeholder_with_default(0., shape=(), name='input_dropout'),
            'num_features_nonzero': tf.placeholder(tf.int32, name='input_zero')  # helper variable for sparse dropout
        }
        self.model()

    def model(self):
        '''
        :param batch_size: 64
        :param encoder_layer:
        :param decoder_layer:
        :param encoder_nodes:
        :param prediction_size:
        :param is_training: True
        :return:
        '''
        p_emd = embedding(self.placeholders['position'], vocab_size=self.para.site_num, num_units=self.para.emb_size,
                          scale=False, scope="position_embed")
        p_emd = tf.reshape(p_emd, shape=[1, self.para.site_num, self.para.emb_size])
        self.p_emd = tf.tile(tf.expand_dims(p_emd, axis=0),
                             [self.para.batch_size, self.para.input_length + self.para.output_length, 1, 1])

        d_emb = embedding(self.placeholders['day'], vocab_size=32, num_units=self.para.emb_size, scale=False,
                          scope="day_embed")
        self.d_emd = tf.reshape(d_emb, shape=[self.para.batch_size, self.para.input_length + self.para.output_length,
                                              self.para.site_num, self.para.emb_size])

        h_emb = embedding(self.placeholders['hour'], vocab_size=24, num_units=self.para.emb_size, scale=False,
                          scope="hour_embed")
        self.h_emd = tf.reshape(h_emb, shape=[self.para.batch_size, self.para.input_length + self.para.output_length,
                                              self.para.site_num, self.para.emb_size])

        m_emb = embedding(self.placeholders['minute'], vocab_size=4, num_units=self.para.emb_size, scale=False,
                          scope="minute_embed")
        self.m_emd = tf.reshape(m_emb, shape=[self.para.batch_size, self.para.input_length + self.para.output_length,
                                              self.para.site_num, self.para.emb_size])

        # encoder
        print('#................................in the encoder step....................................#')
        with tf.variable_scope(name_or_scope='encoder'):
            '''
            return, the gcn output --- for example, inputs.shape is :  (32, 3, 162, 32)
            axis=0: bath size
            axis=1: input data time size
            axis=2: numbers of the nodes
            axis=3: output feature size
            '''
            timestamp = [self.h_emd, self.m_emd]
            position = self.p_emd
            # [-1, input_length, site num, emb_size]
            if self.para.model_name == 'STGIN_1':
                speed = FC(self.placeholders['features_s'], units=[self.para.emb_size, self.para.emb_size],
                           activations=[tf.nn.relu, None],
                           bn=False, bn_decay=0.99, is_training=self.para.is_training)
            else:
                speed = tf.transpose(self.placeholders['features_s'], perm=[0, 2, 1, 3])
                speed = tf.reshape(speed, [-1, self.para.input_length, self.para.features])
                speed3 = tf.layers.conv1d(inputs=speed,
                                          filters=self.para.emb_size,
                                          kernel_size=3,
                                          padding='SAME',
                                          kernel_initializer=tf.truncated_normal_initializer(),
                                          name='conv_1')

                speed2 = tf.layers.conv1d(inputs=tf.reverse(speed, axis=[1]),
                                          filters=self.para.emb_size,
                                          kernel_size=3,
                                          padding='SAME',
                                          kernel_initializer=tf.truncated_normal_initializer(),
                                          name='conv_2')

                speed1 = tf.layers.conv1d(inputs=speed,
                                          filters=self.para.emb_size,
                                          kernel_size=1,
                                          padding='SAME',
                                          kernel_initializer=tf.truncated_normal_initializer(),
                                          name='conv_3')
                # speed2 = tf.nn.sigmoid(speed2)
                speed2 = tf.reverse(speed2, axis=[1])
                speed2 = tf.multiply(speed2, tf.nn.sigmoid(speed2))
                speed3 = tf.multiply(speed3, tf.nn.sigmoid(speed3))
                speed = tf.add_n([speed1, speed2, speed3])
                speed = tf.reshape(speed, [-1, self.para.site_num, self.para.input_length, self.para.emb_size])
                speed = tf.transpose(speed, perm=[0, 2, 1, 3])
            # [-1, input_length, emb_size]
            STE = STEmbedding(position, timestamp, 0, self.para.emb_size, False, 0.99, self.para.is_training)

            encoder = Encoder_ST(hp=self.para, placeholders=self.placeholders, model_func=self.model_func)
            encoder_outs = encoder.encoder_spatio_temporal(speed=speed,
                                                           STE=STE[:, :self.para.input_length, :, :],
                                                           supports=self.supports)
            print('encoder encoder_outs shape is : ', encoder_outs.shape)

        # bridge
        print('#................................in the bridge step.....................................#')
        with tf.variable_scope(name_or_scope='bridge'):
            # STE[:, :self.para.input_length,:,:]
            # bridge_outs = transformAttention(encoder_outs, encoder_outs, STE[:, self.para.input_length:,:,:], self.para.num_heads, self.para.emb_size // self.para.num_heads, False, 0.99, self.para.is_training)
            bridge = BridgeTransformer(self.para)
            bridge_outs = bridge.encoder(X=encoder_outs,
                                         X_P=encoder_outs,
                                         X_Q=STE[:, self.para.input_length:, :, :])
            print('bridge bridge_outs shape is : ', bridge_outs.shape)

        # decoder
        # print('#................................in the decoder step....................................#')
        # with tf.variable_scope(name_or_scope='decoder'):
        #     '''
        #     return, the gcn output --- for example, inputs.shape is :  (32, 1, 162, 32)
        #     axis=0: bath size
        #     axis=1: input data time size
        #     axis=2: numbers of the nodes
        #     axis=3: output feature size
        #     '''
        #     decoder = Decoder_ST(hp=self.para, placeholders=self.placeholders, model_func=self.model_func)
        #     decoder_outs = decoder.decoder_spatio_temporal(speed=bridge_outs,
        #                                                    STE = STE[:, self.para.input_length:,:,:],
        #                                                    supports=self.supports)
        #     print('decoder decoder_outs shape is : ', decoder_outs.shape)

        # inference
        print('#................................in the inference step...................................#')
        with tf.variable_scope(name_or_scope='inference'):
            inference = InferenceClass(para=self.para)
            self.pres_s = inference.inference(out_hiddens=bridge_outs)
            print('pres_s shape is : ', self.pres_s.shape)

        self.loss1 = tf.reduce_mean(
            tf.sqrt(tf.reduce_mean(tf.square(self.pres_s + 1e-10 - self.placeholders['labels_s']), axis=0)))
        self.train_op_1 = tf.train.AdamOptimizer(self.para.learning_rate).minimize(self.loss1)

        print('#...............................in the training step.....................................#')

    def test(self):
        '''
        :param batch_size: usually use 1
        :param encoder_layer:
        :param decoder_layer:
        :param encoder_nodes:
        :param prediction_size:
        :param is_training: False
        :return:
        '''
        model_file = tf.train.latest_checkpoint('weights/')
        self.saver.restore(self.sess, model_file)

    def initialize_session(self):
        self.sess = tf.Session()
        self.saver = tf.train.Saver(var_list=tf.trainable_variables())

    def re_current(self, a, max, min):
        return [num * (max - min) + min for num in a]

    def run_epoch(self):
        '''
        from now on,the model begin to training, until the epoch to 100
        '''
        start_time = datetime.datetime.now()
        max_mae = 100
        self.sess.run(tf.global_variables_initializer())
        iterate = DataClass(self.para)

        train_next = iterate.next_batch(batch_size=self.para.batch_size, epoch=self.para.epoch, is_training=True)

        for i in range(int((iterate.length // self.para.site_num * self.para.divide_ratio - (
                self.para.input_length + self.para.output_length)) // self.para.step)
                       * self.para.epoch // self.para.batch_size):
            x_s, day, hour, minute, label_s, x_p, label_p = self.sess.run(train_next)
            x_s = np.reshape(x_s, [-1, self.para.input_length, self.para.site_num, self.para.features])
            day = np.reshape(day, [-1, self.para.site_num])
            hour = np.reshape(hour, [-1, self.para.site_num])
            minute = np.reshape(minute, [-1, self.para.site_num])
            feed_dict = construct_feed_dict(x_s, self.adj, label_s, day, hour, minute, x_p, label_p, self.placeholders)
            feed_dict.update({self.placeholders['dropout']: self.para.dropout})

            loss_1, _ = self.sess.run((self.loss1, self.train_op_1), feed_dict=feed_dict)
            print("after %d steps,the training average loss value is : %.6f" % (i, loss_1))

            # validate processing
            if i % 100 == 0:
                mae = self.evaluate()
                if max_mae > mae:
                    print("the validate average loss value is : %.6f" % (mae))
                    max_mae = mae
                    self.saver.save(self.sess, save_path=self.para.save_path + 'model.ckpt')
        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        print("Total running times is : %f" % total_time.total_seconds())

    def evaluate(self):
        '''
        :param para:
        :param pre_model:
        :return:
        '''
        label_s_list, pre_s_list = list(), list()

        # with tf.Session() as sess:
        model_file = tf.train.latest_checkpoint(self.para.save_path)
        if not self.para.is_training:
            print('the model weights has been loaded:')
            self.saver.restore(self.sess, model_file)
            # self.saver.save(self.sess, save_path='gcn/model/' + 'model.ckpt')

        iterate_test = DataClass(hp=self.para)
        test_next = iterate_test.next_batch(batch_size=self.para.batch_size, epoch=1, is_training=False)
        max_s, min_s = iterate_test.max_s['speed'], iterate_test.min_s['speed']

        # '''
        for i in range(int((iterate_test.length // self.para.site_num
                            - iterate_test.length // self.para.site_num * iterate_test.divide_ratio
                            - (
                                    self.para.input_length + self.para.output_length)) // iterate_test.output_length) // self.para.batch_size):
            x_s, day, hour, minute, label_s, x_p, label_p = self.sess.run(test_next)
            x_s = np.reshape(x_s, [-1, self.para.input_length, self.para.site_num, self.para.features])
            day = np.reshape(day, [-1, self.para.site_num])
            hour = np.reshape(hour, [-1, self.para.site_num])
            minute = np.reshape(minute, [-1, self.para.site_num])
            feed_dict = construct_feed_dict(x_s, self.adj, label_s, day, hour, minute, x_p, label_p, self.placeholders)
            feed_dict.update({self.placeholders['dropout']: 0.0})

            pre_s = self.sess.run((self.pres_s), feed_dict=feed_dict)
            label_s_list.append(label_s)
            pre_s_list.append(pre_s)

        label_s_list = np.reshape(np.array(label_s_list, dtype=np.float32),
                                  [-1, self.para.site_num, self.para.output_length]).transpose([1, 0, 2])
        pre_s_list = np.reshape(np.array(pre_s_list, dtype=np.float32),
                                [-1, self.para.site_num, self.para.output_length]).transpose([1, 0, 2])
        if self.para.normalize:
            label_s_list = np.array(
                [self.re_current(np.reshape(site_label, [-1]), max_s, min_s) for site_label in label_s_list])
            pre_s_list = np.array(
                [self.re_current(np.reshape(site_label, [-1]), max_s, min_s) for site_label in pre_s_list])
        else:
            label_s_list = np.array([np.reshape(site_label, [-1]) for site_label in label_s_list])
            pre_s_list = np.array([np.reshape(site_label, [-1]) for site_label in pre_s_list])
        print('speed prediction result')
        label_s_list = np.reshape(label_s_list, [-1])
        pre_s_list = np.reshape(pre_s_list, [-1])
        mae, rmse, mape, cor, r2 = metric(label_s_list, pre_s_list)  # 产生预测指标
        # describe(label_list, predict_list)   #预测值可视化
        return mae