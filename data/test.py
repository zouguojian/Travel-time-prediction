# -- coding: utf-8 --

import pandas as pd
import datetime
data = pd.read_csv('trajectory_1.csv',encoding='utf-8')
print(max(data['speed_1'].values()))
time = data.values[1][2]
print(time)

time = datetime.datetime.strptime(time,'%Y-%m-%d %H:%M:%S')

print(time)
print(time.minute)

print(sum([1,2,3,4]))

if time<datetime.datetime.now():
    print((datetime.datetime.now()-time).total_seconds()//60)