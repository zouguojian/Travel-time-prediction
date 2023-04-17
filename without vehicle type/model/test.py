# -- coding: utf-8 --
import datetime

# 6/13/2021 10:05:05 AM

a = [2,3,5,6,2,4,1,0,3,2,1,2,3,2,1,2]

s = dict()

i=0
for char in set(a):
       s[char] =i
       i+=1

print(s)