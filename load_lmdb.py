# -*- coding: utf-8 -*-
import scipy as sp
import numpy as np
import lmdb
import feat_helper_pb2
import time

start_time = time.time()
lmdb_name = 'data84_resnet_features'

if 'db' not in locals().keys():
    db = lmdb.open(lmdb_name)
    txn= db.begin()
    cursor = txn.cursor()
    cursor.iternext()
    datum = feat_helper_pb2.Datum()
    keys = []
    values = []
    for key, value in enumerate( cursor.iternext_nodup()):
        keys.append(key)
        values.append(cursor.value())
num = len(values)
dim = len(value[1])
print 'number_samples',num
print 'number_features',dim

A = np.zeros((num,dim))
for im_idx in range(num):
    datum.ParseFromString(values[im_idx])
    A[im_idx, :] = datum.float_data
np.savetxt('data84_resnet_features.txt',A)


print 'the end,time spent: ' ,time.time() - start_time
