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
---

### Experimental Results
#### HA

#### MT-STAN