# -- coding: utf-8 --
import tensorflow as tf
import numpy as np
import argparse
from model.hyparameter import parameter
import pandas as pd
import datetime

dragon_dragon={('78000F', '780011'): 28, ('780011', '780013'): 29, ('780011', '78005D'): 30, ('780013', '780019'): 31,
               ('780019', '78001B'): 32, ('78001B', '78001D'): 33, ('78001B', '780079'): 34, ('78001D', '78001F'): 35,
               ('78001F', '780021'): 36, ('780021', '780023'): 37, ('78005D', '78005F'): 38, ('78005F', '780061'): 39,
               ('780061', '780063'): 40, ('780061', '78007B'): 41, ('780061', '79007A'): 42, ('780063', '780021'): 43,
               ('780067', '780069'): 44, ('780069', '78006B'): 45, ('780079', '780063'): 46, ('780079', '78007B'): 47,
               ('780079', '790062'): 48, ('78007B', '780067'): 49, ('78007B', '78007D'): 50, ('78007D', '78007F'): 51,
               ('790012', '790010'): 52, ('790014', '790012'): 53, ('79001A', '790014'): 54, ('79001C', '79001A'): 55,
               ('79001E', '780079'): 56, ('79001E', '79001C'): 57, ('790020', '79001E'): 58, ('790022', '790020'): 59,
               ('790022', '790064'): 60, ('790024', '790022'): 61, ('79005E', '790012'): 62, ('790060', '79005E'): 63,
               ('790062', '790060'): 64, ('790064', '78007B'): 65, ('790064', '790062'): 66, ('790064', '79007A'): 67,
               ('790068', '78007D'): 68, ('790068', '79007C'): 69, ('79006A', '790068'): 70, ('79006C', '79006A'): 71,
               ('79007A', '78001D'): 72, ('79007A', '79001C'): 73, ('79007C', '780063'): 74, ('79007C', '790062'): 75,
               ('79007C', '79007A'): 76, ('79007E', '780067'): 77, ('79007E', '79007C'): 78, ('790080', '79007E'): 79}
route = [('780019', '78001B'),('78001B', '78001D'), ('78001D', '78001F'), ('78001F', '780021'), ('780021', '780023')]

class DataClass(object):
    def __init__(self, hp=None):
        '''
        :param hp:
        '''
        self.hp = hp                              # hyperparameter
        self.min_value=0.000000000001
        self.input_length=self.hp.input_length         # time series length of input
        self.output_length=self.hp.output_length       # the length of prediction
        self.is_training=self.hp.is_training           # true or false
        self.divide_ratio=self.hp.divide_ratio         # the divide between in training set and test set ratio
        self.step=self.hp.step                         # windows step
        self.site_num=self.hp.site_num
        self.trajectory_length=self.hp.trajectory_length
        self.file_train_s= self.hp.file_train_s
        self.file_train_p = self.hp.file_train_p
        self.normalize = self.hp.normalize             # data normalization

        self.data_s=self.get_source_data(self.file_train_s)
        self.shape_s=self.data_s.shape
        self.data_tra=self.get_source_data(self.file_train_p)
        self.shape_p=self.data_tra.shape

        self.vehicle_id={} # store the index from vehicle id to index
        self.length=self.data_s.shape[0]                        # data length
        self.max_s, self.min_s= self.get_max_min(self.data_s)   # max and min values' dictionary
        self.max_p, self.min_p = self.get_max_min(self.data_tra)  # max and min values' dictionary

        self.normalization(self.data_s, ['speed'], max_dict=self.max_s, min_dict=self.min_s, is_normalize=self.normalize)                  # normalization
        self.normalization(self.data_tra, list(self.data_tra.keys())[4:], max_dict=self.max_p, min_dict=self.min_p, is_normalize=self.normalize)  # normalization

    def get_source_data(self,file_path=None):
        '''
        :return:
        '''
        data = pd.read_csv(file_path, encoding='utf-8')
        return data

    def get_max_min(self, data=None, times =1):
        '''
        :param data:
        :return:
        '''
        min_dict=dict()
        max_dict=dict()

        for key in data.keys():
            min_dict[key] = data[key].min()
            max_dict[key] = data[key].max()
        # print('the max feature list is :', max_dict)
        # print('the min feature list is :', min_dict)
        return max_dict, min_dict

    def normalization(self, data, keys=None, max_dict =None, min_dict=None, is_normalize=True):
        '''
        :param data:
        :param keys:  is a list
        :param is_normalize:
        :return:
        '''
        if is_normalize:
            for key in keys:
                data[key]=(data[key] - min_dict[key]) / (max_dict[key] - min_dict[key] + self.min_value)

    def generator(self):
        '''
        speed: [length, site num, feature]
        week: [length, site num]
        day: [length, site num]
        hour: [length, site num]
        minute: [length, site num]
        label: [site num, output length]

        vehicle ID: [1]
        vehicle type: [1]
        start week: [1]
        start day: [1]
        start hour: [1]
        start minute: [1]
        start second: [1]
        distance: [trajectory length]
        separate trajectory ID: [trajectory_length]
        label separate time: [trajectory_length]
        label total time: [1]
        :return:
        '''
        data_s = self.data_s.values
        data_tra = self.data_tra.values
        if self.is_training:
            low, high = 0, int(self.shape_s[0]//self.site_num * self.divide_ratio)
        else:
            low, high = int(self.shape_s[0]//self.site_num * self.divide_ratio), int(self.shape_s[0]//self.site_num)
        speed_low, speed_high =self.input_length, int(self.shape_s[0]//self.site_num * self.divide_ratio)

        while speed_low + self.input_length + self.output_length <= speed_high:
            if datetime.datetime.strptime(data_tra[low, 2],'%Y-%m-%d %H:%M:%S') >= data_s[speed_low, -1] and datetime.datetime.strptime(data_tra[low, 2],'%Y-%m-%d %H:%M:%S') < data_s[speed_low+1, -1]:
                # 个体轨迹数据与交通速度数据之间的对应
                label=data_s[speed_low * self.site_num: (speed_low + self.output_length) * self.site_num, -1:]
                label=np.concatenate([label[i * self.site_num : (i + 1) * self.site_num, :] for i in range(self.output_length)], axis=1)

                yield (data_s[(speed_low -self.input_length) * self.site_num : speed_low * self.site_num, 5:6],                         # speed
                       data_s[(speed_low -self.input_length) * self.site_num : (speed_low + self.output_length) * self.site_num, 2]//7, # week
                       data_s[(speed_low -self.input_length) * self.site_num : (speed_low + self.output_length) * self.site_num, 2],    # day
                       data_s[(speed_low -self.input_length) * self.site_num : (speed_low + self.output_length) * self.site_num, 3],    # hour
                       data_s[(speed_low -self.input_length) * self.site_num : (speed_low + self.output_length) * self.site_num, 4]//15,# minute
                       label, # speed label

                       self.vehicle_id[data_tra[low,0]], # vehicle id
                       data_tra[low, 1],                 # vehicle type
                       datetime.datetime.strptime(data_tra[low, 2], '%Y-%m-%d %H:%M:%S').day//7, # start week
                       datetime.datetime.strptime(data_tra[low, 2], '%Y-%m-%d %H:%M:%S').day,    # start day
                       datetime.datetime.strptime(data_tra[low, 2], '%Y-%m-%d %H:%M:%S').hour,   # start hour
                       datetime.datetime.strptime(data_tra[low, 2], '%Y-%m-%d %H:%M:%S').minute, # start minute
                       datetime.datetime.strptime(data_tra[low, 2], '%Y-%m-%d %H:%M:%S').second, # start second
                       np.array([data_tra[low, 4 + i * 4] for i in range(self.trajectory_length)],dtype=np.float),       # distances
                       np.array([dragon_dragon[tuple] for tuple in route],dtype=np.int),                                 # route id
                       np.array([data_tra[low, 5 + i * 4] for i in range(self.trajectory_length)], dtype=np.float),      # separate trajectory time
                       np.array(sum([data_tra[low, 5 + i * 4] for i in range(self.trajectory_length)]), dtype=np.float)) # total time
                low += 1
            else:
                speed_low+=1

    def next_batch(self, batch_size, epoch, is_training=True):
        '''
        :param batch_size:
        :param epochs:
        :param is_training:
        :return: x shape is [batch, input_length*site_num, features];
                 day shape is [batch, (input_length+output_length)*site_num];
                 hour shape is [batch, (input_length+output_length)*site_num];
                 minute shape is [batch, (input_length+output_length)*site_num];
                 label shape is [batch, output_length*site_num, features]
        '''

        self.is_training=is_training
        dataset=tf.data.Dataset.from_generator(self.generator,output_types=(tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.float32,  # speed
                                                                            tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32,
                                                                            tf.float32, tf.float32))

        if self.is_training:
            dataset=dataset.shuffle(buffer_size=int(self.shape_s[0]//self.hp.site_num * self.divide_ratio-self.input_length-self.output_length)//self.step)
            dataset=dataset.repeat(count=epoch)
        dataset=dataset.batch(batch_size=batch_size)
        iterator=dataset.make_one_shot_iterator()

        return iterator.get_next()
#
if __name__=='__main__':
    para = parameter(argparse.ArgumentParser())
    para = para.get_para()

    iter=DataClass(hp=para)
    print(iter.data_s.keys())

    next=iter.next_batch(batch_size=12, epoch=1, is_training=False)
    with tf.Session() as sess:
        for _ in range(4):
            x, d, h, m, y=sess.run(next)
            print(x.shape)
            print(y.shape)
            print(d[0,0],h[0,0],m[0,0])