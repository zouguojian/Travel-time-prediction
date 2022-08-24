# -- coding: utf-8 --
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import argparse
import datetime

from model.embedding import embedding
# from model.deepfm import DeepFM
from model.trajectory_inference import DeepFM
from model.hyparameter import parameter
from model.data_next import DataClass

tf.reset_default_graph()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Model(object):
    def __init__(self, hp):
        self.hp = hp
        self.site_num = self.hp.site_num
        self.input_length = self.hp.input_length
        self.output_length = self.hp.output_length
        self.emb_size = self.hp.emb_size
        self.batch_size = self.hp.batch_size
        self.feature_s = self.hp.feature_s
        self.feature_tra = self.hp.feature_tra
        self.learning_rate = self.hp.learning_rate
        self.field_cnt = self.hp.field_cnt
        self.epoch = self.hp.epoch
        self.divide_ratio = self.hp.divide_ratio
        self.step = self.hp.step
        self.dropout = self.hp.dropout

        self.initial_placeholder()
        self.initial_speed_embedding()
        self.model()

    def initial_placeholder(self):
        # define placeholders
        self.placeholders = {
            'position': tf.placeholder(tf.int32, shape=(1, self.site_num), name='input_position'),
            'week': tf.placeholder(tf.int32, shape=(None, self.site_num), name='input_week'),
            'day': tf.placeholder(tf.int32, shape=(None, self.site_num), name='input_day'),
            'hour': tf.placeholder(tf.int32, shape=(None, self.site_num), name='input_hour'),
            'minute': tf.placeholder(tf.int32, shape=(None, self.site_num), name='input_minute'),
            'feature_s': tf.placeholder(tf.float32, shape=[None, self.input_length, self.site_num, self.feature_s], name='input_s'),
            'labels_s': tf.placeholder(tf.float32, shape=[None, self.site_num, self.output_length], name='labels_s'),
            'feature_tra': tf.placeholder(tf.float32, shape=[None, self.feature_tra], name='input_tra'),
            'labels_tra': tf.placeholder(tf.float32, shape=[None, self.output_length], name='labels_tra'),
            'labels_tra_sum': tf.placeholder(tf.float32, shape=[None, 1], name='labels_tra_sum'),
            'feature_inds': tf.placeholder(dtype=tf.int32, shape=[None, self.field_cnt], name='feature_inds'),
            'dropout': tf.placeholder_with_default(0., shape=(), name='input_dropout')
        }

    def initial_speed_embedding(self):
        # speed related embedding define
        p_emd = embedding(self.placeholders['position'], vocab_size=self.site_num, num_units=self.emb_size, scale=False, scope="position_embed")
        self.p_emd = tf.tile(tf.expand_dims(p_emd, axis=0), [self.batch_size, self.input_length + self.output_length, 1, 1])

        w_emb = embedding(self.placeholders['week'], vocab_size=5, num_units=self.emb_size, scale=False, scope="week_embed")
        self.w_emd = tf.reshape(w_emb, shape=[self.batch_size, self.input_length + self.output_length, self.site_num, self.emb_size])

        d_emb = embedding(self.placeholders['day'], vocab_size=32, num_units=self.emb_size, scale=False, scope="day_embed")
        self.d_emd = tf.reshape(d_emb, shape=[self.batch_size, self.input_length + self.output_length, self.site_num, self.emb_size])

        h_emb = embedding(self.placeholders['hour'], vocab_size=24, num_units=self.emb_size, scale=False, scope="hour_embed")
        self.h_emd = tf.reshape(h_emb, shape=[self.batch_size, self.input_length + self.output_length, self.site_num, self.emb_size])

        m_emb = embedding(self.placeholders['minute'], vocab_size=4, num_units=self.emb_size, scale=False, scope="minute_embed")
        self.m_emd = tf.reshape(m_emb, shape=[self.batch_size, self.input_length + self.output_length, self.site_num, self.emb_size])

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
        print('#................................feature cross....................................#')
        with tf.variable_scope(name_or_scope='trajectory_model'):
            DeepModel = DeepFM(self.hp)
            self.pre = DeepModel.inference(X=self.placeholders['feature_tra'],
                                            feature_inds=self.placeholders['feature_inds'],
                                            keep_prob=self.placeholders['dropout'])

        self.loss = tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(self.pre + 1e-10 - self.placeholders['labels_tra']), axis=0)))
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

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
        iterate = DataClass(self.hp)
        train_next = iterate.next_batch(batch_size=self.batch_size, epoch=self.epoch, is_training=True)

        for i in range(int((iterate.length // self.site_num * self.para.divide_ratio - (
                self.input_length + self.output_length)) // self.para.step)
                       * self.epoch // self.batch_size):
            x_s, day, hour, minute, label_s, x_p, label_p = self.sess.run(train_next)
            x_s = np.reshape(x_s, [-1, self.input_length, self.site_num, self.feature_s])
            day = np.reshape(day, [-1, self.site_num])
            hour = np.reshape(hour, [-1, self.site_num])
            minute = np.reshape(minute, [-1, self.site_num])
            feed_dict = construct_feed_dict(x_s, label_s, day, hour, minute, x_p, label_p, self.placeholders)
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
                                  [-1, self.site_num, self.output_length]).transpose([1, 0, 2])
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