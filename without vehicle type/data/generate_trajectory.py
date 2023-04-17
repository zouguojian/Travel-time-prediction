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

keys = ['entry_id', 'exit_id', 'vehicle_id', 'vehicle_type', 'start_time',
       'end_time', 'distance', 'travel_time', 'speed']
keychar='end_time_@,distance_@,travel_time_@,speed_@,'

def generate_tra(file_name_list=None, pre_filename='data_list/', target_file='trajectory_1.csv'):
    # file_name_list [('78000F', '780011'), ('780011', '78005D'),...,]
    write_file = open(target_file, 'w', encoding='utf-8')
    writer = csv.writer(write_file)
    writer.writerow(keys[2:5] + ''.join([keychar.replace('@', str(i+1)) for i in range(len(file_name_list))]).strip(',').split(','))
    dict_roads = {dragon_dragon[file_name]:{'start_sets': {}, 'end_sets': {}} for file_name in file_name_list} # 存储每个路段的行驶车辆数据
    '''
    dict_roads: {
                    28: {
                            'start_sets': {
                                            '宁A1897C_0 + start_time': ['vehicle_id', 'vehicle_type', 'start_time',
                                                                    'end_time', 'distance', 'travel_time', 'speed']
                                        }
                            'end_sets': {
                                            '宁A1897C_0 + end_time': ['vehicle_id', 'vehicle_type', 'start_time',
                                                                    'end_time', 'distance', 'travel_time', 'speed']
                                      }
                        }
                }
    '''
    # 每个路段的行车记录进入到字典中，进行存储，从而实现毫秒级相应查找
    for file_name in file_name_list:
        data = pd.read_csv(pre_filename + str(dragon_dragon[file_name]) + '.csv', encoding='utf-8')
        data['key_date'] = pd.to_datetime(data.end_time)
        data = data.sort_values(by='key_date', axis=0, ascending=True) # 按照时间的顺序进行升序排序
        for line in data.values:
            dict_roads[dragon_dragon[file_name]]['start_sets'][line[2]+ '+' +line[4]] = line[:9]  # 车牌+开始时间的字典
            # dict_roads[dragon_dragon[file_name]]['end_sets'][line[2] + '+' + line[5]] = line[2:9] # 车牌+结束时间的字典
    print('# data load finished #')

    for key in dict_roads[dragon_dragon[file_name_list[0]]]['start_sets']:
        new_whole_trajectory=list(dict_roads[dragon_dragon[file_name_list[0]]]['start_sets'][key][:9])
        key_1 = key # 短路径开始 key = '车牌+开始时间的字典'
        for index in range(1, len(file_name_list)):
            # print(key_1.split('+'), dict_roads[dragon_dragon[file_name_list[index-1]]]['start_sets'][key_1][5])
            key_2 = key_1.split('+')[0] + '+' + dict_roads[dragon_dragon[file_name_list[index-1]]]['start_sets'][key_1][5]
            # 下一个短路径开始 key = '车牌+开始时间的字典'，即为上一个短路径的结束 '车牌+结束时间的字典'
            if key_2 in dict_roads[dragon_dragon[file_name_list[index]]]['start_sets']: # 判断是否在当前路径的开始字典中
                new_whole_trajectory+=list(dict_roads[dragon_dragon[file_name_list[index]]]['start_sets'][key_2][5:9])
            else: break
            key_1 = key_2
        if (len(new_whole_trajectory)-9) // 4 == len(file_name_list)-1:
            print(new_whole_trajectory)
            writer.writerow(new_whole_trajectory[2:])
    write_file.close()

file_name_list =[('780019', '78001B'),('78001B', '78001D'), ('78001D', '78001F'), ('78001F', '780021'), ('780021', '780023')]
generate_tra(file_name_list=file_name_list,target_file='trajectory_1.csv')
# keys = ['entry_id', 'exit_id', 'vehicle_id', 'vehicle_type', 'start_time',
#        'end_time', 'distance', 'travel_time', 'speed', 'key_date']