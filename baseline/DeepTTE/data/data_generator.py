# -- coding: utf-8 --
import pandas as pd
import datetime
import pickle
import json

route_1= {1:[106.3369005,38.51238957],2:[106.3350869,38.48446767],3:[106.3105628,38.41086277],4:[106.288999,38.3862921],
          5:[106.2373622,38.30641865],6:[106.1630849,38.11581089]}

def get_source_data(file_path=None): # obtain the source data
    '''
    :return:
    '''
    data = pd.read_csv(file_path, encoding='utf-8')
    return data


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

def generator(file=None, to_file=None, is_training=True, divide_ratio=0.8, la_ln={}):
    '''
    :param file:
    :param to_file:
    :return:
    '''
    """
    driverID   # vehicle ID
    weekID     # week the day of week, from 0 to 6 (Mon to Sun)
    timeID     # the ID of the start time (in minute), from 0 to 1439
    dateID     # the date in a month, from 0 to 30
    
    lats       # the sequence of longitutes of all sampled GPS points
    lngs       # the sequence of latitudes of all sampled GPS points
    dist_gap   # the same length as lngs
    states     # the sequence of taxi states (available/unavaible). we can use 1 ro replace, if you dont care this features
    
    time_gap   # the same length as lngs
    dist       # total distance of the path (KM)
    time       # total travel time (min)
    """

    data = get_source_data(file_path=file)
    data = data.values
    vehicle_id = get_vehicle_id(data[:, 0])
    print(vehicle_id)

    with open(to_file, 'w', newline='\n') as wri:
        if is_training:
            low, high = 0, int(data.shape[0] * divide_ratio)
        else:
            low, high = int(data.shape[0] * divide_ratio), int(data.shape[0])
        while low < high:
            line = {}
            line['driverID'] = vehicle_id[data[low, 0]]
            line['weekID'] = (datetime.datetime.strptime(data[low, 2], '%Y-%m-%d %H:%M:%S').day-1)%7
            line['timeID'] = datetime.datetime.strptime(data[low, 2], '%Y-%m-%d %H:%M:%S').hour * 60 + datetime.datetime.strptime(data[low, 2], '%Y-%m-%d %H:%M:%S').minute
            line['dateID'] = datetime.datetime.strptime(data[low, 2], '%Y-%m-%d %H:%M:%S').day-1

            line['lats'] = [la_ln[i+1][0] for i in range(len(la_ln))]
            line['lngs'] = [la_ln[i+1][1] for i in range(len(la_ln))]
            line['dist_gap'] = [0]+[sum([data[low, 4 + j * 4] for j in range(i+1)])/1000.0 for i in range(len(la_ln)-1)]
            line['states'] =[1.0 for _ in range(len(la_ln))]

            line['time_gap'] = [0.] + [sum([data[low, 5 + i * 4] for j in range(i+1)]) * 3600 for i in range(len(la_ln)-1)]
            line['dist'] = sum([data[low, 4 + i * 4] for i in range(len(la_ln)-1)])/1000.0
            line['time'] = sum([data[low, 5 + i * 4] for i in range(len(la_ln)-1)]) * 60.0

            print(line)
            # wri.write(str(line)+'\n')
            json.dump(line, wri)
            wri.write('\n')

            low+=1
    wri.close()
    return

print('beginning')
# generator(file='trajectory_1.csv', to_file='train_1', is_training=True, divide_ratio=0.8, la_ln=route_1)

data = get_source_data(file_path='trajectory_1.csv')
data = data.values
vehicle_id = get_vehicle_id(data[:, 0])
print(len(vehicle_id))

print('finished')