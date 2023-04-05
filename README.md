# TRAVEL-TIME-PREDICTION

>In this paper, we proposed a novel travel time prediction model, named MT-STAN. We use multi-tricks
 to solve the problems we meet in existing researches, for example, individual travel preferences may 
 affect total travel time. We give the detail experimental results and all models' parameters and weights, 
 including MT-STAN and baselines.
 ---
 
## WHAT SHOULD WE PAY ATTENTION TO FOCUS ON THE RUNNING ENVIRONMENT?

<font face="微软雅黑" >Note that we need to install the right packages to guarantee the model runs according to the file requirements.txt！！！</font>
  
>* first, please use the conda tool to create a virtual environment, such as ‘conda create traffic speed’;  
> * second, active the environment, and conda activate traffic speed;   
> * third, build environment, the required environments have been added in the file named requirements.txt; you can use conda as an auxiliary tool to install or pip, e.g., conda install tensorflow==1.13.1;    
> * if you have installed the last TensorFlow version, it’s okay; import tensorflow.compat.v1 as tf and tf.disable_v2_behavior();    
> * finally, please click the run_train.py file; the codes are then running;  
> * Note that our TensorFlow version is 1.14.1 and can also be operated on the 2.0. version.  
---

## EXPERIMENTAL SETTING
> Our MT-STAN model’s hyperparameters and baselines are optimised during training by picking the model with the lowest mean absolute error (MAE) from the validation set. Therefore, the model error determined by the validation set is used to choose the best model to use. The procedure entails the following details: 50 epochs are used in each experiment. The model is put to the test on the validation set once it has been trained for an epoch. We revise and save the model parameters if the MAE of the prediction model drops on the validation set. Once the prediction impact of the prediction model on the validation set is maximized after extensive experimentation with various parameter settings, the training procedure is complete. The test set samples are then iterated until a prediction is reached. All of our studies use an early- stop mechanism, with parameters of 300 early-stop rounds and 10 maximum epochs. Hyperparameters were optimized during training by picking the model with the lowest mean absolute error (MAE) from the validation set. 

> For example, if we need to define the number of heads for temporal attention, other hyperparameters should be fixed, such as blocks. We change the number of heads from 1 to d and observe the prediction model’s performance on the validation set. The machine will stop the training processing and save the parameter weights if the MAE does not change within the 300 rounds of the training phase and 10 maximum epochs. Finally, we select the optimal hyperparameter with the best performance on the validation dataset. In addition, the most challenging is to set the values of the hyperparameters $\lambda$ and $\eta$ for multi-task learning because we set $\lambda$ as a float, and several experiments and attempts (attempts more than 50 times) will be taken to choose an appropriate value for $\eta$. The attempts are the same as the previous, according to MAE on the validation dataset. To consist with existing studies, we decided on a value of 6 for the objective time step Q and a value of 12 for the historical time step P. Seventy percent of the information was used as a training set, 15 percent as a validation set, and 15 percent as a test set in the experiment.

> The final model framework parameters are established after many training stages. The MT-STAN model’s layer count, node count, and other relevant hyperparameters are listed in Table I of manuscript. The MT-STAN and baselines are implemented in TensorFlow and PyTorch. The server’s 4 NVIDIA Tesla V100S- PCIE-32GB GPUs and 24 CPU cores are used for model training and testing. It is worth noting that both the suggested MT-STAN model and the baseline models’ implementation codes are freely accessible on the author’s GitHub.

## NOTED THE DATASET CHANGES
* some of places need to be modified, including hyparameter.py and data_next.py  
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
---

## MT-STAN AND BASELINES （ALL METHODS' CODES HAVE BEEN REPRODUCED） 
#### CoDriver ETA  [codes link](https://github.com/zouguojian/Travel-time-prediction/tree/main/baseline/CoDriverETA)
#### DeepTTE [codes link](https://github.com/zouguojian/Travel-time-prediction/tree/main/baseline/DeepTTE)
#### WDR [codes link](https://github.com/zouguojian/Travel-time-prediction/tree/main/baseline/WDR)
#### CompactETA [codes link](https://github.com/zouguojian/Travel-time-prediction/tree/main/baseline/CompactETA)
#### CTTE [codes link](https://github.com/zouguojian/Travel-time-prediction/tree/main/baseline/CTTE)
#### MT-STAN [codes link](https://github.com/zouguojian/Travel-time-prediction)
---

## EXPERIMENTAL RESULTS


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


## Vehicle Type (-[standard-link](http://jt.hlj.gov.cn/gip/ewebeditor/uploadfile/20201106155214231.pdf))

* The vehicle type as following,  

|Vehicle Type |Definition	   |
|  ----       | ----    | 
|1类客车(1-一型客车)	   |Passenger cars that a length is less than 6000 mm and a seating capacity is no more than 9 people        |
|2类客车(2-二型客车)	   |Passenger cars that a length is less than 6000 mm and a seating capacity is 10-19 people                 |
|3类客车(3-三型客车)	   |Passenger cars that a length is no less than 6000 mm and a seating capacity is no more than 39 people    |
|4类客车(4-四型客车)	   |Passenger cars that the length is no less than 6000 mm and a seating capacity is no more than 40 people	 |
|1类货车(11-一型货车)	   |Truck that the number of suspension shaft is 2, and the length is less than 6000 mm, and the total weight is not less than 4500kg  |
|2类货车(12-二型货车)	   |Truck that the number of suspension shaft is 2, and the length is less than 6000 mm, or the total weight is not less than 4500kg	   |
|3类货车(13-三型货车)	   |Truck that the number of suspension shaft is 3|
|4类货车(14-四型货车)	   |Truck that the number of suspension shaft is 4|
|5类货车(15-五型货车)	   |Truck that the number of suspension shaft is 5|
|6类货车(16-六型货车)	   |Truck that the number of suspension shaft is 6|
|1类专项作业车(21-一型专项作业车)	   |Operation Van that the number of suspension shaft is 2, and the length is less than 6000 mm, and the total weight is not less than 4500kg |
|2类专项作业车(22-二型专项作业车)	   |Operation Van that the number of suspension shaft is 2, and the length is less than 6000 mm, or the total weight is not less than 4500kg |
|3类专项作业车(23-三型专项作业车)	   |Operation Van that the number of suspension shaft is 3|
|4类专项作业车(24-四型专项作业车)	   |Operation Van that the number of suspension shaft is 4|
|5类专项作业车(25-五型专项作业车)	   |Operation Van that the number of suspension shaft is 5|
|6类专项作业车(26-六型专项作业车)	   |Operation Van that the number of suspension shaft is no less than 6|