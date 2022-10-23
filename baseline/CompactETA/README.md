# CompactETA

* In this paper, we develop a novel ETA learning system named as CompactETA, 
which provides an accurate online travel time inference within 100 microseconds. 
In the proposed method, we encode high order spatial and tem- poral dependency into 
sophisticated representations by applying graph attention network on a spatiotemporal 
weighted road net- work graph. We further encode the sequential information of the 
travel route by positional encoding to avoid the recurrent network structure. The 
properly learnt representations enable us to apply a very simple multi-layer perceptron 
model for online real-time inference. 
   
   
* 需要改动的地方分别为:hyparameter.py和data_next.py   
>self.parser.add_argument('--save_path', type=str, default='weights/CompactETA-2/', help='save path')  
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


