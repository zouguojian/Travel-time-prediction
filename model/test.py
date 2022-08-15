# -- coding: utf-8 --
import datetime

# 6/13/2021 10:05:05 AM

keys = ['entry_id', 'exit_id', 'vehicle_id', 'vehicle_type', 'start_time',
       'end_time', 'distance', 'travel_time', 'speed']
keychar='end_time_@,distance_@,travel_time_@,speed_@'
print(keys[2:5] + ''.join([keychar.replace('@', str(i+1)) for i in range(5)]).split(','))