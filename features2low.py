# -*- coding: utf-8 -*-
from matplotlib import pyplot
import scipy as sp
import numpy as np
import lmdb
import feat_helper_pb2
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import time

start_time = time.time()


def load_lmdb(lmdb_name,dim):
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
    print 'number of samples: ',num
    A = np.zeros((num,dim))
    for im_idx in range(num):
        datum.ParseFromString(values[im_idx])
        A[im_idx, :] = datum.float_data
    return A,num

#read data
rawdata,num   = load_lmdb('../ctr/lr/lrdata/test_features',4096)
#rawdata,num   = load_lmdb('sub50_features',4096)
#rawdata = np.array([[-1, -1, 1], [-2, -1, 4], [-3, -2 ,1], [1, 1 ,2], [2, 1 ,1], [3, 2 ,3]])
x = np.nan_to_num(rawdata)
#x_train, x_test = train_test_split(x,test_size = 0.1)
#print np.nan_to_num(a)

pca = PCA(n_components = 512)
x_low = pca.fit_transform(x)
#test_low = transform(x_test)
#np.savetxt('lowfeatures.txt',lowfeatures)
#np.savetxt('variance_ratio.txt',pca.explained_variance_ratio_)
#print 'score: ',pca.score(x)
#print 'precision: ',pca.get_precision()
#print 'ratio: ',pca.explained_variance_ratio_

neigh_x = NearestNeighbors(n_neighbors=4)
neigh_x.fit(x)
x_neighbors = neigh_x.kneighbors(x,return_distance=False)

neigh_low = NearestNeighbors(n_neighbors=4)
neigh_low.fit(x_low)
xlow_neighbors =neigh_low.kneighbors(x_low,return_distance=False)
x_xlow = np.hstack((x_neighbors,xlow_neighbors))

pos_num = 0
top1_negnum = 0
for i in range(num):
    j = 1
#    print x_neighbors[i][j],xlow_neighbors[i][j]
    while j<4 and x_neighbors[i][j] == xlow_neighbors[i][j]:
        j +=1
#    print j
    if j==1:
        top1_negnum +=1
    elif j==4:
        pos_num+=1
top1_accscore = float(num-top1_negnum)/num
acc_score = float(pos_num)/num
print 'top1_accscore= ',top1_accscore
print 'pos_num= ',pos_num
print 'acc_score: ',acc_score

np.savetxt('3neigh_compare.txt',x_xlow)
print 'the end of pca,time spent: ' ,time.time() - start_time
