## read train/test/validation data and transformation bases
## transform data into wanted structures
## return user and item number, and padding train data

import json
import numpy as np
import random as rd
import math

def read_data(path,batch_size):
    with open(path) as f:
        line = f.readline()
        data = json.loads(line)
    f.close()
    user_num = len(data)
    item_num = 0
    interactions = []
    for user in range(user_num):
        for item in data[user]:
            interactions.append((user, item))
            item_num = max(item, item_num)
    item_num += 1
    return(data, interactions, user_num, item_num)

def read_bases(path):
    with open(path) as f:
        line = f.readline()
        bases = json.loads(line)
    f.close()
    [feat_u, feat_v] = bases
    feat_u = np.array(feat_u).astype(np.float32)
    feat_v = np.array(feat_v).astype(np.float32)
    return [feat_u, feat_v]

def normalize(features):
    [f_u, f_v] = features
    f_u_norm = np.linalg.norm(f_u, axis=1, keepdims=True)
    f_v_norm = np.linalg.norm(f_v, axis=1, keepdims=True)
    f_u_norm = np.mean(f_u_norm)
    f_v_norm = np.mean(f_v_norm)
    return [f_u/f_u_norm, f_v/f_v_norm]
