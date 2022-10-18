This project is the code of AAAI 2018 paper ***When Will You Arrive? Estimating Travel Time Based on Deep Neural Networks***.

几点注意：class Net(nn.Module): 中的 'driverID'需要根据每个数据集进行改变。  
confg.json文件需要的内容，已经放在本页下面  
main函数中的内容需要根据实际情况进行改变  

# Usage:

## Model Training
python train.py
### Parameters:

* task: train/test
* batch_size: the batch_size to train, default 400
* epochs: the epoch to train, default 100
* kernel_size: the kernel size of Geo-Conv, only used when the model contains the Geo-conv part
* pooling_method: attention/mean
* alpha: the weight of combination in multi-task learning
* log_file: the path of log file
* result_file: the path to save the predict result. By default, this switch is off during the training

Example:
```
python main.py --task train  --batch_size 10  --result_file ./result/deeptte.res --pooling_method attention --kernel_size 3 --alpha 0.1 --log_file run_log
```


## Model Evaluation

### Parameters:
* weight_file: the path of model weight
* result_file: the path to save the result

## Example:
```
python main.py --task test --weight_file ./saved_weights/weight --batch_size 10  --result_file ./result/deeptte.res --pooling_method attention --kernel_size 3 --alpha 0.1
```

## How to User Your Own Data
In the data folder we provide some sample data. You can use your own data with the corresponding format as in the data samples. The sampled data contains 1800 trajectories. To make the model performance close to our proposed result, make sure your dataset contains more than 5M trajectories.

### Format Instructions
Each sample is a json string. The key contains:
* driverID
* dateID: the date in a month, from 0 to 30
* weekID: the day of week, from 0 to 6 (Mon to Sun)
* timeID: the ID of the start time (in minute), from 0 to 1439
* dist: total distance of the path (KM)
* time: total travel time (min), i.e., the ground truth. You can set it as any value during the test phase
* lngs: the sequence of longitutes of all sampled GPS points
* lats: the sequence of latitudes of all sampled GPS points
* states: the sequence of taxi states (available/unavaible). You can remove this attributes if it is not available in your dataset. See models/base/Attr.py for details.
* time_gap: the same length as lngs. Each value indicates the time gap from current point to the firt point (set it as arbitrary values during the test)
* dist_gap: the same as time_gap

The GPS points in a path should be resampled with nearly equal distance.

Furthermore, repalce the config file according to your own data, including the dist_mean, time_mean, lngs_mean, etc.

* route_1 (number of driver id is 'driverID', 30224)  
    "dist_gap_mean": 9.59178799999548,  
    "dist_gap_std": 6.929101195820786,  
    "time_gap_mean": 383.94512807028354,  
    "time_gap_std": 777.4550845532493,  
    "lngs_mean": 106.27866605000001,  
    "lngs_std":  0.061595111085666114,  
    "lats_mean": 38.36937360833333,  
    "lats_std": 0.13163407178348177,  
    "dist_mean": 47.95894000001783,  
    "dist_std": 1.7834622667578515e-11,  
    "time_mean":  1919.7256403514048,  
    "time_std": 1646.7810113131948,  
    "train_set": ["train_1"],  
    "eval_set": ["test_1"],  
    "test_set": ["test_1"]  
    
* route_2 (number of driver id is 'driverID', 31971)
    "dist_gap_mean": 10.176746000000037,  
    "dist_gap_std": 6.840308228161815,  
    "time_gap_mean": 519.9825320266333,  
    "time_gap_std": 1785.8106934603047,  
    "lngs_mean": 106.31036083333333,  
    "lngs_std":  0.08210806022331295,  
    "lats_mean": 38.38040202166666,  
    "lats_std": 0.1377800106073268,  
    "dist_mean": 50.88372999994004,  
    "dist_std": 1,  
    "time_mean":  2599.912660133196,  
    "time_std": 3853.486207250185,  
    "train_set": ["train_2"],  
    "eval_set": ["test_2"],  
    "test_set": ["test_2"]  
    
    
* route_3 (number of driver id is 'driverID', 1075)
    "dist_gap_mean": 9.942610000000682,  
    "dist_gap_std": 7.975530792999024,  
    "time_gap_mean": 507.0262301953788,  
    "time_gap_std": 527.4411675335208,  
    "lngs_mean": 106.383451325,  
    "lngs_std":  0.03404479343022875,  
    "lats_mean": 38.3021282275,  
    "lats_std": 0.08310905639061898,  
    "dist_mean": 29.827830000000212,  
    "dist_std": 1,  
    "time_mean":  1521.0786905861482,  
    "time_std": 402.32395561272597,  
    "train_set": ["train_3"],  
    "eval_set": ["test_3"],  
    "test_set": ["test_3"]   
    
    
* route_4 (number of driver id is 'driverID', 9979)  
    "dist_gap_mean": 5.9273379999866,  
    "dist_gap_std": 2.471477353095,  
    "time_gap_mean": 273.22975292259736,  
    "time_gap_std": 717.5542593619709,  
    "lngs_mean": 106.39431156666666,  
    "lngs_std":  0.06540195880103691,  
    "lats_mean": 38.415134855,  
    "lats_std": 0.08673265357470158,  
    "dist_mean": 29.636690000005864,  
    "dist_std": 1,  
    "time_mean":  1366.1487646129779,  
    "time_std": 1602.0938382640356,  
    "train_set": ["train_4"],  
    "eval_set": ["test_4"],  
    "test_set": ["test_4"]  