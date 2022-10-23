# WDR

* WDR model has three main blocks: 1) the wide model is similar to the wide model 
in Wide & Deep network. We use a second order cross-product transformation followed
 by an affine transformation to get a 256 dimensional output; 2) the deep model embeds
  the sparse features into a 20 dimensional space, then processes the concatenated features
   by a 3-hidden-layer MLP with a ReLU activation to get a 256 dimensional output. 
   The size of all the three hidden layers in the MLP is 256; 3) the recurrent model is a 
   variant of standard RNN. The feature of each road segment is first projected into a 256 
   dimensional space by a fully connected layer with ReLU as the activation function. The 
   transformed feature is then fed into to a standard LSTM with cell size 256. 
   
   
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


