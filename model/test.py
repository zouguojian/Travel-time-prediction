# -- coding: utf-8 --

import numpy as np

a = np.random.random([32, 12, 200])

print(a.shape)

print(a[:,[2,3,4,5]].shape)

print([1]+[6, 4, 3])