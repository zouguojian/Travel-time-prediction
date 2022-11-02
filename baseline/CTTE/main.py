# -- coding: utf-8 --
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import argparse
import datetime
import csv
from baseline.CTTE.model.hyparameter import parameter
from baseline.CTTE.model.ctte_inf import CTTEClass
from baseline.CTTE.model.data_next import DataClass
from baseline.CTTE.model.utils import construct_feed_dict, one_hot_concatenation, metric, FC, SE

tf.reset_default_graph()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logs_path = "board"

class Model(object):
    def __init__(self, hp, SE_=None):
        self.hp = hp
        self.step = self.hp.step  # window length
        self.epoch = self.hp.epoch  # total training epochs
        self.dropout = self.hp.dropout  # dropout
        self.site_num = self.hp.site_num  # number of roads
        self.emb_size = self.hp.emb_size  # hidden embedding size
        self.is_training = self.hp.is_training
        self.field_cnt = self.hp.field_cnt  # number of features fields
        self.feature_s = self.hp.feature_s  # number of speed features
        self.batch_size = self.hp.batch_size  # batch size
        self.feature_tra = self.hp.feature_tra  # number of trajectory features
        self.divide_ratio = self.hp.divide_ratio  # the ratio of training set
        self.input_length = self.hp.input_length  # input length of speed data
        self.output_length = self.hp.output_length  # output length of speed data
        self.learning_rate = self.hp.learning_rate  # learning rate
        self.trajectory_length = self.hp.trajectory_length  # trajectory length
        self.SE_=SE_   # spatial embedding
        self.initial_placeholder()
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
            'label_s': tf.placeholder(tf.float32, shape=[None, self.site_num, self.output_length], name='label_s'),
            'feature_tra': tf.placeholder(tf.float32, shape=[None, self.feature_tra], name='input_tra'),
            'label_tra': tf.placeholder(tf.float32, shape=[None, self.trajectory_length], name='label_tra'),
            'label_tra_sum': tf.placeholder(tf.float32, shape=[None, 1], name='label_tra_sum'),
            'feature_inds': tf.placeholder(dtype=tf.int32, shape=[None, self.field_cnt], name='feature_inds'),
            'trajectory_inds': tf.placeholder(dtype=tf.int32, shape=[self.trajectory_length], name='feature_inds'),
            'se': tf.placeholder(dtype=tf.float32, shape=[self.site_num, self.emb_size], name='spatial_embedding_for_road_network'),
            'dropout': tf.placeholder_with_default(0., shape=(), name='input_dropout')
        }

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
        print('#................................model loading....................................#')

        speed = FC(self.placeholders['feature_s'], units=[self.emb_size, self.emb_size],
                   activations=[tf.nn.relu, None],
                   bn=False, bn_decay=0.99, is_training=self.is_training)
        speed = tf.gather(speed, indices=self.placeholders['trajectory_inds'], axis=2)  # (32, 24, 5, 64)

        SE_ = tf.gather(self.placeholders['se'], indices=self.placeholders['trajectory_inds'])  # (trajectory length, 64)

        with tf.variable_scope(name_or_scope='trajectory_model'):
            CTTEModel = CTTEClass(self.hp)
            self.pre_s, self.pre_t = CTTEModel.inference(speed=speed[:, :self.input_length], # (32, 12, 5, 64)
                                                         feature_inds=self.placeholders['feature_inds'],
                                                         keep_prob=self.placeholders['dropout'],
                                                         SE_=SE_)
            print(self.pre_s.shape, self.pre_t.shape)

        self.pre_s_o = tf.gather(self.placeholders['label_s'], indices=self.placeholders['trajectory_inds'], axis=1)
        self.loss_1 = tf.losses.mean_squared_error(labels=self.pre_s, predictions=self.pre_s_o)

        mae = tf.losses.absolute_difference(self.pre_t, self.placeholders['label_tra_sum'])
        mape = tf.divide(mae, self.placeholders['label_tra_sum'])
        self.loss_2 = tf.reduce_mean(mape)

        self.loss = self.loss_1 + 0.3 * self.loss_2

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        print('#...............................in the training step...............................#')

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
        return a * (max - min) + min

    def run_epoch(self):
        '''
        from now on,the model begin to training, until the epoch to 100
        '''
        start_time = datetime.datetime.now()
        max_mae = 100
        self.sess.run(tf.global_variables_initializer())
        iterate = DataClass(self.hp)
        train_next = iterate.next_batch(batch_size=self.batch_size, epoch=self.epoch, is_training=True)

        for i in range(int(iterate.shape_tra[0] * self.divide_ratio) * self.epoch // self.batch_size):
            x_s, week, day, hour, minute, label_s, \
            vehicle_id, vehicle_type, start_week, start_day, start_hour, start_minute, start_second, distances, route_id, \
            element_index, separate_trajectory_time, total_time, trajectory_inds,_,_,_,_ = self.sess.run(train_next)

            x_s = np.reshape(x_s, [-1, self.input_length, self.site_num, self.feature_s])
            week = np.reshape(week, [-1, self.site_num])
            day = np.reshape(day, [-1, self.site_num])
            hour = np.reshape(hour, [-1, self.site_num])
            minute = np.reshape(minute, [-1, self.site_num])
            x_tra = one_hot_concatenation(
                features=[vehicle_id, vehicle_type, start_week, start_day, start_hour, start_minute, start_second,
                          distances, route_id])
            feed_dict = construct_feed_dict(x_s=x_s,
                                            week=week,
                                            day=day,
                                            hour=hour,
                                            minute=minute,
                                            label_s=label_s,
                                            x_tra=x_tra,
                                            element_index=element_index,
                                            separate_trajectory_time=separate_trajectory_time,
                                            total_time=total_time,
                                            trajectory_inds=trajectory_inds,
                                            placeholders=self.placeholders,
                                            se=self.SE_)
            feed_dict.update({self.placeholders['dropout']: self.dropout})

            loss, _ = self.sess.run((self.loss, self.train_op), feed_dict=feed_dict)
            # loss1, _ = self.sess.run((self.loss1, self.train_op), feed_dict=feed_dict)
            # loss, _ = self.sess.run((self.loss2, self.train_op), feed_dict=feed_dict)
            print("after %d steps,the training average loss value is : %.6f" % (i, loss))

            # validate processing
            if i % 100 == 0:
                mae = self.evaluate()
                if max_mae > mae:
                    print("the validate average loss value is : %.6f" % (mae))
                    max_mae = mae
                    self.saver.save(self.sess, save_path=self.hp.save_path + 'model.ckpt')
        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        print("Total running times is : %f" % total_time.total_seconds())

    def evaluate(self):
        '''
        :param para:
        :param pre_model:
        :return:
        '''
        label_tra_sum_list, pre_tra_sum_list = list(), list()

        # with tf.Session() as sess:
        model_file = tf.train.latest_checkpoint(self.hp.save_path)
        if not self.hp.is_training:
            print('the model weights has been loaded:')
            self.saver.restore(self.sess, model_file)

        iterate_test = DataClass(hp=self.hp)
        test_next = iterate_test.next_batch(batch_size=self.batch_size, epoch=1, is_training=False)
        max_s, min_s = iterate_test.max_s['speed'], iterate_test.min_s['speed']

        file = open('/Users/guojianzou/Travel-time-prediction/results/'+str(self.hp.model_name)+'-4'+'.csv', 'w', encoding='utf-8')
        writer = csv.writer(file)
        writer.writerow(['vehicle_id', 'vehicle_type', 'time', 'whether_app', 'pre_sum', 'label_sum'])

        for i in range(int(iterate_test.shape_tra[0] * (1 - self.hp.divide_ratio) - 15 * (
                self.input_length + self.output_length)) // self.batch_size):
            x_s, week, day, hour, minute, label_s, \
            vehicle_id, vehicle_type, start_week, start_day, start_hour, start_minute, start_second, distances, route_id, \
            element_index, separate_trajectory_time, total_time, trajectory_inds, dates, vehicle_id_str, vehicle_type_int, whether_app = self.sess.run(test_next)
            x_s = np.reshape(x_s, [-1, self.input_length, self.site_num, self.feature_s])
            week = np.reshape(week, [-1, self.site_num])
            day = np.reshape(day, [-1, self.site_num])
            hour = np.reshape(hour, [-1, self.site_num])
            minute = np.reshape(minute, [-1, self.site_num])
            x_tra = one_hot_concatenation(
                features=[vehicle_id, vehicle_type, start_week, start_day, start_hour, start_minute, start_second,
                          distances, route_id])
            feed_dict = construct_feed_dict(x_s=x_s,
                                            week=week,
                                            day=day,
                                            hour=hour, minute=minute,
                                            label_s=label_s,
                                            x_tra=x_tra,
                                            element_index=element_index,
                                            separate_trajectory_time=separate_trajectory_time,
                                            total_time=total_time,
                                            trajectory_inds=trajectory_inds,
                                            placeholders=self.placeholders,
                                            se=self.SE_)
            feed_dict.update({self.placeholders['dropout']: 0.0})
            y = self.sess.run((self.pre_t), feed_dict=feed_dict)

            print([vehicle_id_str[0].decode(), vehicle_type_int, dates[0], whether_app, y[0], total_time[0]])
            writer.writerow([vehicle_id_str[0].decode(), vehicle_type_int[0], dates[0], whether_app[0], y[0,0] * 60, total_time[0,0] * 60])

            # print(dates, pre_tra_sum * 60, total_time * 60)
            label_tra_sum_list.append(total_time)
            pre_tra_sum_list.append(y)

        label_tra_sum_list = np.reshape(np.array(label_tra_sum_list, dtype=np.float32) * 60, [-1, 1])  # total trajectory travel time for label
        pre_tra_sum_list = np.reshape(np.array(pre_tra_sum_list, dtype=np.float32) * 60, [-1, 1])  # total trajectory travel time for prediction

        print('entire travel time prediction result >>>')
        mae_tra_sum, rmse_tra_sum, mape_tra_sum, cor_tra_sum, r2_tra_sum = metric(pred=pre_tra_sum_list, label=label_tra_sum_list)  # 产生预测指标
        # describe(label_list, predict_list)   #预测值可视化
        return mae_tra_sum

def main(argv=None):
    '''
    :param argv:
    :return:
    '''
    print('#.....................................beginning.....................................#')
    para = parameter(argparse.ArgumentParser())
    para = para.get_para()

    print('Please input a number : 1 or 0. (1 and 0 represents the training or testing, respectively).')
    val = input('please input the number : ')

    if int(val) == 1:
        para.is_training = True
    else:
        para.batch_size = 1
        para.is_training = False
    SE_ = SE(para.file_SE)
    pre_model = Model(para, SE_=SE_)
    pre_model.initialize_session()

    if int(val) == 1:
        pre_model.run_epoch()
    else:
        pre_model.evaluate()

    print('#..................................finished.........................................#')


if __name__ == '__main__':
    main()