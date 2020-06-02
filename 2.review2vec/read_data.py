## read train/test/validation data and transformation bases
## transform data into wanted structures
## return user and item number, and padding train data
## Wenhui Yu 2020.06.02
## author @Wenhui Yu, yuwh16@mails.tsinghua.edu.cn

import json
import numpy as np

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

def load_data(path):
    with open(path) as f:
        line = f.readline()
        data = json.loads(line)
    f.close()
    return data

def load_features(path):
    with open(path) as f:
        line = f.readline()
        features = json.loads(line)
    f.close()
    return np.array(features).astype(np.float32)

def read_features(path):
    with open(path) as f:
        line = f.readline()
        [features1, features2] = json.loads(line)
    f.close()
    return np.array(features1).astype(np.float32), np.array(features2).astype(np.float32)



