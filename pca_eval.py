# -*- coding: utf-8 -*-
import scipy as sp
import numpy as np
import time

start_time = time.time()

#read data
neighs   = np.loadtxt('negh_compare1.txt')
num = len(neighs)
sum_acc3 = 0
for i in range(num):
    a3_list = neighs[i][1:4]
    b3_list = neighs[i][5:8]
    inter3_list = list((set(a3_list).union(set(b3_list)))^(set(a3_list)^set(b3_list))
    acc3 = float(len(inter3_list))/len(a3_list)
    sum_acc3 += acc3
recall_top3 = sum_acc3/num
print 'recall_top3: ',recall_top3


