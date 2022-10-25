# CoDriverETA

* Estimated time of arrival (ETA) is one of the most important services in 
intelligent transportation systems (ITS). Precise ETA ensures proper travel 
scheduling of passengers as well as guarantees efficient decision-making on 
ride-hailing platforms, which are used by an explosively growing number of 
people in the past few years. Recently, machine learning-based methods have 
been widely adopted to solve this time estimation problem and become state-of-the-art. 
However, they do not well explore the personalization information, as many 
drivers are short of personalized data and do not have sufficient trajectory 
data in real applications. This data sparsity problem prevents existing methods 
from obtaining higher prediction accuracy. In this article, we propose a novel 
deep learning method to solve this problem. We introduce an auxiliary task to 
learn an embedding of the personalized driving information under multi-task 
learning framework. In this task, we discriminatively learn the embedding of 
driving preference that preserves the historical statistics of driving speed. 
For this purpose, we adapt the triplet network from face recognition to learn 
the embedding by constructing triplets in the feature space. This simultaneously 
learned embedding can effectively boost the prediction accuracy of the travel time.   
   
   
* 需要改动的地方分别为:hyparameter.py和data_next.py  
>self.parser.add_argument('--save_path', type=str, default='weights/WDR-2/', help='save path')  
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


