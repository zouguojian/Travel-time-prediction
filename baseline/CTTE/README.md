# CTTE

* In this paper, we propose Customized Travel Time Estimation (CTTE) 
that fuses GPS trajectories, smartphone inertial data, and road network 
within a deep recurrent neural network. It constructs a road link traffic 
database with topology representation, speed statistics, and query 
distribution. It also calibrates inertial readings, estimates the arbitrary 
phone’s pose in car, and detects multiple aggressive driving events (e.g., 
bump judders, sharp turns, sharp slopes, frequent lane shifts, overspeeds, 
and sudden brakes). Finally, we demonstrate our solution on two typical 
transportation problems, i.e., predicting traffic speed at holistic level 
and estimating customized travel time at personal level, within a multi-task 
learning structure. 

#### Note  
> In this paper, the data from the real time phone data to calculate the driving behavior. However, in the 
highway network, it is difficult to get the phone data, and the task is that we need to estimate the travel time 
previously according to the departure time and related features, and the whole travel data is come from tolls. 
Therefore, for driver in highway network, to predict the travel time before we departure maybe it is not 
available use phone data. In our WORK, we need to note that we replace the original driving behavior USE 
Global preference features (departure time, vehicle ID, vehicle type, etc), and the data is from real world 
data of highway network. 
   
   
* 需要改动的地方分别为:hyparameter.py和data_next.py   
>self.parser.add_argument('--save_path', type=str, default='weights/CTTE-2/', help='save path')  
self.parser.add_argument('--field_cnt', type=int, default=17, help='the number of filed for trajectory features')  
>self.parser.add_argument('--feature_tra', type=int, default=30542, help='number of the trajectory feature elements')  
>self.parser.add_argument('--trajectory_length', type=int, default=5, help='length of trajectory')  
>self.parser.add_argument('--file_train_t', type=str, default='/Users/guojianzou/Travel-time-prediction/data/trajectory_2.csv', help='trajectory file address')  

>route = [('780019', '78001B'),('78001B', '78001D'), ('78001D', '78001F'), ('78001F', '780021'), ('780021', '780023')]  
>max_road_leangth = 22193.94  

>data1-feature_tra: 30542  
>data2-feature_tra: 32274  
>data3-feature_tra: 1385  
>data4-feature_tra: 10284  


