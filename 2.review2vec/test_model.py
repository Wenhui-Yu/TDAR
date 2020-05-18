## test the model and return the performance

from evaluation import *
from read_data import *
from params import DIR
from params import TOP_K
from params import TEST_VALIDATION
import random as rd
import numpy as np
import operator
import gc

train_path = DIR+'train_data.json'
teat_path = DIR+'test_data.json'
validation_path = DIR+'validation_data.json'

## load data
[train_data, train_data_interaction, user_num, item_num] = read_data(train_path,1)
teat_vali_path = validation_path if operator.eq(TEST_VALIDATION,'Validation')==1 else teat_path
test_data = read_data(teat_vali_path,1)[0]

def test_one_user(x):
    k_num = len(TOP_K)
    f1 = np.zeros(k_num)
    ndcg = np.zeros(k_num)
    user = x[0]
    score = x[1]
    score_min = min(score) - 1
    for item in train_data[user]:
        score[item] = score_min
    order = list(np.argsort(score))
    order.reverse()
    for i in range(k_num):
        f1[i] += evaluation_F1(order, TOP_K[i], test_data[user])
        ndcg[i] += evaluation_NDCG(order, TOP_K[i], test_data[user])
    return f1, ndcg

def test_model(sess, model, user_review_feature, item_review_feature):
    user_batch = np.array(range(user_num))
    item_batch = np.array(range(item_num))
    result = []
    try:
        all_ratings = sess.run(model.all_ratings,
                               feed_dict={model.users: user_batch,
                                          model.items: item_batch,
                                          model.user_word: user_review_feature,
                                          model.item_word: item_review_feature})
    except:
        all_ratings = sess.run(model.all_ratings,
                               feed_dict={model.users: user_batch,
                                          model.items: item_batch})
    for user in user_batch:
        if len(test_data[user]) > 0:
            result.append(test_one_user([user, all_ratings[user]]))
    result = np.array(result)
    F1, NDCG = np.mean(np.array(result), axis=0)
    del result, all_ratings
    gc.collect()
    return F1, NDCG
    

