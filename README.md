# Travel-time-prediction

>In this paper, we proposed a novel travel time prediction model, named MT-STAN. We use multi-tricks
 to solve the problems we meet in existing researches, for example, individual travel preferences may 
 affect total travel time. We give the detail experimental results and all models' parameters and weights, 
 including MT-STAN and baselines.
 
## Precautions

<font face="微软雅黑" >需要注意的是，需要根据requirements.txt文件中指示的包进行安装，才能正常的运行程序！！！</font>
>* 首先，使用conda创建一个虚拟环境，如‘conda create travel_time_prediction'  
> * 激活环境，conda activate travel_time_prediction；  
> * 安装环境，需要安装的环境已经添加在requirements.txt中，可以用conda安装，也可以使用pip安装，如：conda install tensorflow==1.12.0；  
> * 如果安装的是最新的tensorflow环境，没问题，tensorflow的包按照以下方式进行导入即可：import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()；  
> * 点击 main.py文件即可运行代码。
> * 需要注意的是，我们在tensorflow的1.12和1.14版本环境中都可以运行，更高版本运行添加import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()部分代码即可正常运行

## Noted the Dataset Changes
* 需要改动的地方分别为:hyparameter.py和data_next.py  
>self.parser.add_argument('--save_path', type=str, default='weights/MT-STAN-1/', help='save path')  
self.parser.add_argument('--field_cnt', type=int, default=17, help='the number of filed for trajectory features')  
>self.parser.add_argument('--feature_tra', type=int, default=30542, help='number of the trajectory feature elements')  
>self.parser.add_argument('--trajectory_length', type=int, default=5, help='length of trajectory')  
>self.parser.add_argument('--file_train_t', type=str, default='/Users/guojianzou/Travel-time-prediction/data/trajectory_1.csv', help='trajectory file address')  

>route = [('780019', '78001B'),('78001B', '78001D'), ('78001D', '78001F'), ('78001F', '780021'), ('780021', '780023')]  
>max_road_leangth = 22193.94  

>data1-feature_tra: 30542  
>data2-feature_tra: 32274  
>data3-feature_tra: 1385  
>data4-feature_tra: 10284  

# MT-STAN and Baselines （all methods' codes have been reproduced） 
#### CoDriver ETA  [codes link](https://github.com/zouguojian/Travel-time-prediction/tree/main/baseline/CoDriverETA)
#### DeepTTE [codes link](https://github.com/zouguojian/Travel-time-prediction/tree/main/baseline/DeepTTE)
#### WDR [codes link](https://github.com/zouguojian/Travel-time-prediction/tree/main/baseline/WDR)
#### CompactETA [codes link](https://github.com/zouguojian/Travel-time-prediction/tree/main/baseline/CompactETA)
#### CTTE [codes link](https://github.com/zouguojian/Travel-time-prediction/tree/main/baseline/CTTE)
#### MT-STAN [codes link](https://github.com/zouguojian/Travel-time-prediction)

# Experimental Results


|Fuyin Expressway (G70)|  Jingzang Expressway (G6) |  YinKun Expressway (G85)	|
|  ----                |  ----                     |  ----                      |

|Model           |MAE	   | RMSE	 |MAPE     |MAE	     | RMSE	   |MAPE     |MAE	   | RMSE	 |MAPE    |
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |----     |----    |
|DNN	         |5.187	   |28.663	 |9.376%   |12.089	 |53.375   |14.569%	 |1.256	   |1.679	 |5.434%  |
|CoDriver ETA	 |5.200	   |28.630	 |9.414%   |12.104	 |53.209   |15.145%	 |1.380	   |1.817	 |5.980%  |
|DeepTTE	     |6.108    |29.718	 |12.104%  |13.106	 |53.394   |17.635%	 |3.716	   |5.944	 |13.182% |
|DWR	         |5.172	   |28.666	 |9.250%   |12.059	 |53.126   |14.720%	 |1.406	   |1.827	 |6.171%  |
|CompactETA	     |5.154	   |28.641	 |9.285%   |12.075	 |53.082   |15.010%  |1.236	   |1.620	 |5.341%  |
|CTTE	         |5.191	   |28.617	 |9.384%   |11.963	 |53.182   |14.335%	 |1.271	   |1.960	 |5.195%  |
|Cross-network	 |5.181	   |28.649	 |9.341%   |12.060	 |53.237   |14.725%	 |1.306	   |1.698	 |5.730%  |
|No-Holistic	 |5.146	   |28.618	 |9.239%   |12.032	 |53.301   |14.425%	 |1.250	   |1.683	 |5.484%  |
|No-Multi-Task	 |5.159	   |28.625	 |9.325%   |12.008	 |53.388   |14.285%	 |1.156	   |1.538	 |4.995%  |
|MT-STAN	     |5.130    |28.615	 |9.190%   |11.978	 |53.087   |14.592%	 |1.229	   |1.656	 |5.386%  |


|Qingyin Expressway (G20)|
|  ----                  |

|Model           |MAE	   | RMSE	 |MAPE     |
|  ----          | ----    |  ----   |  ----   |
|DNN	         |4.205	   |28.636	 |9.158%   |
|CoDriver ETA	 |4.236	   |28.724	 |9.077%   |
|DeepTTE	     |5.524	   |26.512	 |18.333%  |
|DWR	         |4.212	   |28.687	 |8.989%   |
|CompactETA	     |4.193	   |28.714	 |8.833%   |
|CTTE	         |4.197	   |28.486	 |8.942%   |
|Cross-network	 |4.222	   |28.702	 |9.065%   |
|No-Holistic	 |4.179	   |28.685	 |8.842%   |
|No-Multi-Task	 |4.181	   |28.705	 |8.765%   |
|MT-STAN	     |4.167	   |28.672	 |8.774%   |


# Vehicle Type [standard](http://jt.hlj.gov.cn/gip/ewebeditor/uploadfile/20201106155214231.pdf)

* 车型分类如下  

|Vehicle Type |definition	   |
|  ----       | ----    | 
|1类客车(1-一型客车)	   |车长小于 6000mm且核定载人数不大于9人的载客汽车 注意，摩托车通行收费公路按１类客车分类|
|2类客车(2-二型客车)	   |核载人数10-19人 车长小于 6000mm且核定载人数为(10-19)人的载客汽车                |
|3类客车(3-三型客车)	   |核载人数≤39人 车长不小于6000mm且核定载人数不大于 39人的载客汽车                  |
|4类客车(4-四型客车)	   |核载人数≥40人 车长不小于6000mm且核定载人数不小于40人的载客汽车	                 |
|1类货车(11-一型货车)	   |总轴数为(含悬浮轴)2 车长小于6000mm且最大允 许总质量小于4500kg      |
|2类货车(12-二型货车)	   |总轴数(含悬浮轴)为2 车长小于6000mm且最大允 许总质量小于4500kg	   |
|3类货车(13-三型货车)	   |总轴数(含悬浮轴)为3	|
|4类货车(14-四型货车)	   |总轴数(含悬浮轴)为4	|
|5类货车(15-五型货车)	   |总轴数(含悬浮轴)为5   |
|6类货车(16-六型货车)	   |总轴数(含悬浮轴)为6   |
|1类专项作业车(21-一型专项作业车)	   |总轴数(含悬浮轴)为2轴 车长小于6000mm且最大允许总质量小于4500kg      |
|2类专项作业车(22-二型专项作业车)	   |总轴数(含悬浮轴)为2轴 车长不小于6000mm且最大允许总质量不小于4500kg   |
|3类专项作业车(23-三型专项作业车)	   |总轴数(含悬浮轴)为3   |
|4类专项作业车(24-四型专项作业车)	   |总轴数(含悬浮轴)为4	  |
|5类专项作业车(25-五型专项作业车)	   |总轴数(含悬浮轴)为5   |
|6类专项作业车(26-六型专项作业车)	   |6类专项作业车的总轴数（含悬浮轴）不小于6轴，大于或等于6轴   |