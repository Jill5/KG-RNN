import collections
import numpy as np
import os
from math import exp
from tensorflow.python import pywrap_tensorflow


def topK(train_data, test_data, user_embs, user_history_dict):
    reader = pywrap_tensorflow.NewCheckpointReader('../model/model.ckpt')  
    var_to_shape_map = reader.get_variable_to_shape_map()   
    item_embs = reader.get_tensor('entity_emb_matrix')

    item_set = set(train_data[:,1]) | set(test_data[:,1])
    n_item = len(item_set)

    tp_dict = dict()		# key为user id，value为用户的真正例list
    user_list = list()		# test data中的user
    for i in range(test_data.shape[0]):
        user = test_data[i,0]
        if test_data[i,2] == 1:
            if user not in tp_dict:
                tp_dict[user] = list()
                user_list.append(user)
            tp_dict[user].append(test_data[i,1])

    top100_list = np.zeros((len(user_list),100))	# 每个用户的top100推荐列表，用户顺序对应user_list
    for u in range(len(user_list)):
        user = user_list[u]
        user_emb = user_embs[user,:]
        pred = np.zeros((n_item,2))
        ind = 0
        for item in item_set:
            pred[ind,0] = item
            if item in user_history_dict[user]:
                pred[ind,1] = 0
            else:
                try:
                    pred[ind,1] = (1 / (1 + exp(- np.dot(user_emb,item_embs[item,:])))) * 5
                except OverflowError:
                    pred[ind,1] = 5
            ind += 1
        pred = pred[pred[:,1].argsort()[::-1]]
        top100_list[u,:] = pred[0:100,0]

    k = [10, 20, 40, 60, 100]
    rec_k = [0, 0, 0, 0, 0]
    pre_k = [0, 0, 0, 0, 0]

    for u in range(len(user_list)):
        user = user_list[u]
        tp_k = [0, 0, 0, 0, 0]	# top k推荐列表中出现的真正例的个数

        for i in range(5):
            for j in range(k[i]):
                if top100_list[u,j] in tp_dict[user]:
                    tp_k[i] += 1
            rec_k[i] += tp_k[i] / len(tp_dict[user])
            pre_k[i] += tp_k[i] / k[i]

    rec_k = [rec/len(user_list) for rec in rec_k]
    pre_k = [pre/len(user_list) for pre in pre_k]

    for i in range(5):
        print('rec@%d: %f	pre@%d: %f' % (k[i], rec_k[i], k[i], pre_k[i]))
