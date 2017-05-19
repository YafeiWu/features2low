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
#rawdata_train,num_train = load_lmdb('../ctr/lr/lrdata/test_features',4096)
rawdata,num   = load_lmdb('sub50_features',4096)
#rawdata_test,num_test = load_lmdb('sub50_features',4096)
dim_low = 256
top = 5
#rawdata,num   = load_lmdb('sub50_features',4096)
#rawdata = np.array([[-1, -1, 1], [-2, -1, 4], [-3, -2 ,1], [1, 1 ,2], [2, 1 ,1], [3, 2 ,3]])
data = np.nan_to_num(rawdata)
#data_test = np.nan_to_num(rawdata_test)
data_train, data_test = train_test_split(data,test_size = 0.1)


def pca_x(x_train,x_test,num,top_n):
    pca = PCA(n_components = num)
    x_low = pca.fit_transform(x_train)
    xtest_low = pca.fit_transform(x_test)
    #test_low = transform(x_test)
    #np.savetxt('lowfeatures.txt',lowfeatures)
    #np.savetxt('variance_ratio.txt',pca.explained_variance_ratio_)
    #print 'score: ',pca.score(x)
    #print 'precision: ',pca.get_precision()
    #print 'ratio: ',pca.explained_variance_ratio_

    neigh_x = NearestNeighbors(n_neighbors=top_n+1)
    neigh_x.fit(x_train)
    xtest_neighbors = neigh_x.kneighbors(x_test,return_distance=False)

    neigh_low = NearestNeighbors(n_neighbors=top_n+1)
    neigh_low.fit(x_low)
    xtestlow_neighbors =neigh_low.kneighbors(xtest_low,return_distance=False)
    x_xlow = np.hstack((xtest_neighbors,xtestlow_neighbors))
    return xtestlow_neighbors,xtestlow_neighbors,x_xlow

test_neigh,test_low_neigh,test_xxlow = pca_x(data_train,data_test,dim_low,top)
#test_neigh,test_low_neigh,test_xxlow = pca_x(data_test,dim_low,top)
#np.savetxt('train_compare.txt',train_xxlow)
np.savetxt('test_compare.txt',test_xxlow)

def pca_eval(x_neighbors,xlow_neighbors,num):
    sum_acc1 = 0
    sum_acc2 = 0
    for i in range(num):
        a_list = x_neighbors[i][1:6]
        a3_list = x_neighbors[i][1:4]
        b_list = xlow_neighbors[i][1:6]
        b3_list = xlow_neighbors[i][1:4]
        
        union_list = list(set(a_list).union(set(b_list)))
        union3_list = list(set(a3_list).union(set(b3_list)))
        inter_list = list((set(a_list).union(set(b_list)))^(set(a_list)^set(b_list)))
        inter3_list = list((set(a3_list).union(set(b3_list)))^(set(a3_list)^set(b3_list)))
    #    acc1 = float(len(inter_list))/len(union_list)
        acc1 = float(len(inter3_list))/len(a3_list)
        acc2 = float(len(inter_list))/len(a_list)
        sum_acc1 += acc1
        sum_acc2 += acc2

    #acc_inter = sum_acc1/num
    recall_top3 = sum_acc1/num
    recall_top5 = sum_acc2/num
    return recall_top3,recall_top5

train_recall_top3,train_recall_top5 = pca_eval(train_neigh,train_low_neigh,num_train)
test_recall_top3,test_recall_top5 = pca_eval(test_neigh,test_low_neigh,num_test)

with open('pca_eval.txt','wb') as pe:
    pe.write('train_recall_top3:'+'\t'+str(train_recall_top3)+'\n')
    pe.write('train_recall_top5:'+'\t'+str(train_recall_top5)+'\n')
    pe.write('test_recall_top3:'+'\t'+str(test_recall_top3)+'\n')
    pe.write('test_recall_top5:'+'\t'+str(test_recall_top5)+'\n')

pe.close

#print 'acc_inter: ',acc_inter
print 'trian_recall_top3: ',train_recall_top3
print 'train_recall_top5: ',train_recall_top5
print 'test_recall_top3: ',test_recall_top3
print 'test_recall_top5: ',test_recall_top5
print 'the end,time spent: ' ,time.time() - start_time
