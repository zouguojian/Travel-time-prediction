# -- coding: utf-8 --

import pandas as pd
import csv

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


keys = ['entry_id', 'exit_id', 'vehicle_id', 'vehicle_type', 'start_time', 'end_time', 'distance', 'travel_time', 'speed']
'''
'entry_id'    : 5, 
'exit_id'     : 1, 
'vehicle_id'  : 3, 
'vehicle_type': 4, 
'start_time'  : 6, 
'end_time'    : 2, 
'distance'    : 7, 
'travel_time' : 8, 
'speed'       : 9
'''

def data_write(file_path, out_path, staion_pair_dict=None, encoding='utf-8'):
    '''
    :param file_path: resource file address
    :param out_path: write path, used to save the training set
    :param staion_pair_dict: what the resource data we want to read
    :return:
    '''
    # using dict to store file and writer address
    file_dict=dict()
    writer_dict=dict()
    for staion_pair in staion_pair_dict.keys():
        file_dict[staion_pair] = open(out_path+str(staion_pair_dict[staion_pair])+'.csv', 'w', encoding=encoding)
        writer_dict[staion_pair] = csv.writer(file_dict[staion_pair])
        writer_dict[staion_pair].writerow(keys)

    reader = open(file=file_path, encoding=encoding)
    reader.readline()
    # read resource file, that is .txt
    for line in reader:
        line = line.strip('\n').split(',')
        print(line)
        writer_dict[(line[5],line[1])].writerow([line[5],line[1],line[3],line[4],line[6],line[2],line[7],line[8],line[9]])

    for staion_pair in dragon_dragon.keys():
        file_dict[staion_pair].close()

# data_write(file_path='/Users/guojianzou/Travel-time-prediction/data/dragon_dragon.txt',out_path='/Users/guojianzou/Travel-time-prediction/data/data_list/',staion_pair_dict=dragon_dragon,encoding='utf-8')