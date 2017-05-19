# -*- coding: utf-8 -*-
import scipy as sp
import numpy as np
import lmdb
import feat_helper_pb2
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import load_files
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
rawdata ,num = load_lmdb('data84_features',4096)
#rawdata,num   = load_lmdb('sub50_features',4096)
dim_low = 512
top = 10
#rawdata,num   = load_lmdb('sub50_features',4096)
data = np.nan_to_num(rawdata)
#data_test = np.nan_to_num(rawdata_test)
#data_train, data_test = train_test_split(data,test_size = 0.1)


def pca_x(x,num,top_n):
    pca = PCA(n_components = num)
    x_low = pca.fit_transform(x)
    #np.savetxt('lowfeatures.txt',lowfeatures)
    #test_low = transform(x_test)
    #np.savetxt('variance_ratio.txt',pca.explained_variance_ratio_)
    #print 'score: ',pca.score(x)
    #print 'precision: ',pca.get_precision()
    #print 'ratio: ',pca.explained_variance_ratio_

    neigh_x = NearestNeighbors(n_neighbors=top_n+1)
    neigh_x.fit(x)
    x_neighbors = neigh_x.kneighbors(x,return_distance=False)

    neigh_low = NearestNeighbors(n_neighbors=top_n+1)
    neigh_low.fit(x_low)
    xlow_neighbors =neigh_low.kneighbors(x_low,return_distance=False)
#    x_xlow = np.hstack((x_neighbors,xlow_neighbors))
    return x_neighbors,xlow_neighbors,x_low

neigh,low_neigh,x_low = pca_x(data,dim_low,top)
np.savetxt('512lowfeatures.txt',x_low)

def pca_eval(x_neighbors,xlow_neighbors,num):
    sum_acc3 = 0
    sum_acc5 = 0
    sum_acc10 = 0
    for i in range(num):
        a10_list = x_neighbors[i][1:11]
        a5_list = x_neighbors[i][1:6]
        a3_list = x_neighbors[i][1:4]
        
        b10_list = xlow_neighbors[i][1:11]
        b5_list = xlow_neighbors[i][1:6]
        b3_list = xlow_neighbors[i][1:4]
        
#        union5_list = list(set(a5_list).union(set(b5_list)))
#        union3_list = list(set(a3_list).union(set(b3_list)))
        inter10_list = list((set(a10_list).union(set(b10_list)))^(set(a10_list)^set(b10_list)))
        inter5_list = list((set(a5_list).union(set(b5_list)))^(set(a5_list)^set(b5_list)))
        inter3_list = list((set(a3_list).union(set(b3_list)))^(set(a3_list)^set(b3_list)))
    #    acc1 = float(len(inter_list))/len(union_list)
    
        acc3 = float(len(inter3_list))/len(a3_list)
        acc5 = float(len(inter5_list))/len(a5_list)
        acc10 = float(len(inter10_list))/len(a10_list)
        sum_acc3 += acc3
        sum_acc5 += acc5
        sum_acc10 += acc10

    #acc_inter = sum_acc1/num
    recall_top3 = sum_acc3/num
    recall_top5 = sum_acc5/num
    recall_top10 = sum_acc10/num
    return recall_top3,recall_top5,recall_top10

recall_top3,recall_top5,recall_top10 = pca_eval(neigh,low_neigh,num)

with open('512pca_eval.txt','wb') as pe:
    pe.write('recall_top3:'+'\t'+str(recall_top3)+'\n')
    pe.write('recall_top5:'+'\t'+str(recall_top5)+'\n')
    pe.write('recall_top10:'+'\t'+str(recall_top10)+'\n')

pe.close

#print 'acc_inter: ',acc_inter
print 'recall_top3: ',recall_top3
print 'recall_top5: ',recall_top5
print 'recall_top10: ',recall_top10
print 'the end,time spent: ' ,time.time() - start_time
