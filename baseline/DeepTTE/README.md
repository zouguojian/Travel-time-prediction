This project is the code of AAAI 2018 paper ***When Will You Arrive? Estimating Travel Time Based on Deep Neural Networks***.

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
    
* route_1 (number of driver id is 'driverID', )