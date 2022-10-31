# -- coding: utf-8 --

import pandas as pd
import numpy as np
import datetime
import csv
da_1 = pd.read_csv('trajectory_1.csv',encoding='utf-8')
da_2 = pd.read_csv('trajectory_2.csv',encoding='utf-8')
print(da_1.shape, da_2.shape)


def verhicle_type_num(data=None):
    vehicle_type = dict()
    for id in data:
        if id not in vehicle_type and id<=16:
            vehicle_type[id] = 1
        elif id<=16: vehicle_type[id] +=1
        else:continue
    return vehicle_type


def get_vehicle_id(data=None):
    '''
    :param data:
    :return:
    '''
    vehicle_id = dict()
    for id in data:
        if id not in vehicle_id:
            vehicle_id[id] = len(vehicle_id)
    return vehicle_id

data_1 = get_vehicle_id(da_1.values[:,0])
data_2 = get_vehicle_id(da_2.values[:,0])
print(len(data_1), len(data_2))

overlap_len=0
for char in data_1:
    if char in data_2: overlap_len+=1

# print('the total length of overlap is : ',overlap_len)
# print(verhicle_type_num(da_1.values[:,1]))
# print(verhicle_type_num(da_2.values[:,1]))

"""
0: 0-5 min
1: 5-10 min
2: 10-15 min
3: 15-20 min
4: 20-25 min
5: 25-30 min
6: 30-35 min
"""
def segment_fuction(data, dataset_index=1, writer=None):
    '''
    :param data:
    :return:
    '''
    # data = pd.read_csv('/Users/guojianzou/Travel-time-prediction/data/statistic/'+str(dataset_index)+'.csv', encoding='utf-8')
    graininess =10
    dict_seg={(i*graininess, (i+1)*graininess):0 for i in range(6)}
    for char in data:
        for i in range(6):
            if i*graininess<char and char<graininess*(i+1):
                dict_seg[((i*graininess, (i+1)*graininess))]+=1
                writer.writerow(['['+str(i*graininess)+', '+str((i+1)*graininess)+')', 'G2-'+str(dataset_index)])
                break
            elif char>60:
                if (60, 10000000) not in dict_seg:
                    dict_seg[(60, 10000000)]=1
                else:
                    dict_seg[(60, 10000000)]+=1
                writer.writerow(['[60, )','G2-'+str(dataset_index)])
                break
                # if (30,)
            else:continue
    print('sum is : ',sum([value for value in dict_seg.values()]))
    return dict_seg

write_file = open('/Users/guojianzou/Travel-time-prediction/data/statistic/1.csv', 'w', encoding='utf-8')
writer = csv.writer(write_file)
writer.writerow(['period', 'road_name'])
print(segment_fuction(da_1.values[:,-2]*60, 1, writer=writer))
print(segment_fuction(da_2.values[:,-2]*60, 2, writer=writer))
write_file.close()