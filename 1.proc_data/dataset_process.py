import random
import json
import re
import random as rd
import numpy as np

def read_data(path):
    interaction = []  # record the interaction
    with open(path) as f:
        for line in f:
            line = eval(line)
            user_id = line['reviewerID']
            item_id = line['asin']
            review_text = set(clean_string(line['reviewText']).split())
            interaction.append([user_id, item_id, review_text])
    return interaction

def write_data(path, data):
    f = open(path, 'w')
    jsObj = json.dumps(data)
    f.write(jsObj)
    f.close()

def dataset_filtering(interaction, core):
    # filtering the dataset with core
    user_id_dic = {}  # record the number of interaction for each user and item
    item_id_dic = {}
    for [user_id, item_id, _] in interaction:
        try:
            user_id_dic[user_id] += 1
        except:
            user_id_dic[user_id] = 1
        try:
            item_id_dic[item_id] += 1
        except:
            item_id_dic[item_id] = 1
    print ('#Original dataset')
    print ('  User:', len(user_id_dic), 'Item:', len(item_id_dic), 'Interaction:', len(interaction), 'Sparsity:', 100 - len(interaction) * 100.0 / len(user_id_dic) / len(item_id_dic), '%')
    sort_user = []
    sort_item = []
    for user_id in user_id_dic:
        sort_user.append((user_id, user_id_dic[user_id]))
    for item_id in item_id_dic:
        sort_item.append((item_id, item_id_dic[item_id]))
    sort_user.sort(key=lambda x: x[1])
    sort_item.sort(key=lambda x: x[1])
    print ('Fitering(core = ', core, '...', end = '')
    while sort_user[0][1] < core or sort_item[0][1] < core:
        # find out all users and items with less than core recorders
        user_LessThanCore = set()
        item_LessThanCore = set()
        for pair in sort_user:
            if pair[1] < core:
                user_LessThanCore.add(pair[0])
            else:
                break
        for pair in sort_item:
            if pair[1] < core:
                item_LessThanCore.add(pair[0])
            else:
                break

        # reconstruct the interaction record, remove the cool one
        interaction_filtered = []
        for [user_id, item_id, text] in interaction:
            if not (user_id in user_LessThanCore or item_id in item_LessThanCore):
                interaction_filtered.append([user_id, item_id, text])
        # update the record
        interaction = interaction_filtered

        # count the number of each user and item in new data, check if all cool users and items are removed
        # reset all memory variables
        user_id_dic = {}  # record the number of interaction for each user and item
        item_id_dic = {}
        for [user_id, item_id, _] in interaction:
            try:
                user_id_dic[user_id] += 1
            except:
                user_id_dic[user_id] = 1
            try:
                item_id_dic[item_id] += 1
            except:
                item_id_dic[item_id] = 1

        sort_user = []
        sort_item = []
        for user_id in user_id_dic:
            sort_user.append((user_id, user_id_dic[user_id]))
        for item_id in item_id_dic:
            sort_item.append((item_id, item_id_dic[item_id]))
        sort_user.sort(key=lambda x: x[1])
        sort_item.sort(key=lambda x: x[1])
        print (len(interaction), end = ' ')
    print()
    print ('#Filtered dataset')
    print ('  User:', len(user_id_dic), 'Item:', len(item_id_dic), 'Interaction:', len(interaction), 'Sparsity:', 100 - len(interaction) * 100.0 / len(user_id_dic) / len(item_id_dic), '%')
    return interaction

def clean_string(string):
    string = re.sub(r"\'s", " is", string)
    string = re.sub(r"\'ve", " have", string)
    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'d", " had", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"[^A-Za-z]", " ", string)
    return string.strip().lower()

def index_encoding(interaction):
    # mapping in into number
    # after filtering the dataset, we need to re-encode the index of users and items
    user_id_set = set()
    item_id_set = set()
    for [user_id, item_id, _] in interaction:
        user_id_set.add(user_id)
        item_id_set.add(item_id)
    user_num2id = list(user_id_set)
    item_num2id = list(item_id_set)
    user_num2id.sort()
    item_num2id.sort()
    user_num = len(user_num2id)
    item_num = len(item_num2id)
    # user_id2num maps id to number, and user_num2id dictionary is not needed, user_ID
    user_id2num = {}
    for num in range(user_num):
        user_id2num[user_num2id[num]] = num
    item_id2num = {}
    for num in range(item_num):
        item_id2num[item_num2id[num]] = num
    interaction_num = []
    user_text = [set() for x in range(user_num)]
    item_text = [set() for x in range(item_num)]
    text = []
    for [user_id, item_id, review_text] in interaction:
        interaction_num.append([user_id2num[user_id], item_id2num[item_id]])
        user_text[user_id2num[user_id]] = user_text[user_id2num[user_id]] | review_text
        item_text[item_id2num[item_id]] = item_text[item_id2num[item_id]] | review_text
        text += list(review_text)
    text = set(text)

    num2word = list(text)
    word2num = {}
    for (i, word) in enumerate(num2word):
        word2num[word] = i
    user_text_num = [[] for x in range(user_num)]
    for u in range(user_num):
        user_text_num[u] = list(map(lambda x: word2num[x], list(user_text[u])))
    item_text_num = [[] for x in range(item_num)]
    for i in range(item_num):
        item_text_num[i] = list(map(lambda x: word2num[x], list(item_text[i])))
    return interaction_num, user_text_num, item_text_num, word2num

def semantic_embedding(word2num, path):
    matrix = np.random.uniform(-1.0, 1.0, (len(word2num), 300))
    with open(path, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode('utf-8', 'ignore')
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in word2num:
                matrix[word2num[word]] = np.fromstring(f.read(binary_len), dtype='float32')
            else: f.read(binary_len)
    return matrix

def dataset_split_dense(Interaction):
    user_interaction = []
    item_interaction = []
    for interaction in Interaction:
        while len(user_interaction) <= interaction[0]:
            user_interaction.append([])
        while len(item_interaction) <= interaction[1]:
            item_interaction.append(0)
        user_interaction[interaction[0]].append(interaction[1])
        item_interaction[interaction[1]] += 1
    validation_data = []
    test_data = []
    for i in range(len(user_interaction)):
        validation_data.append([])
        test_data.append([])
    #print(item_interaction)
    for i in range(len(user_interaction)):
        interactions = user_interaction[i]
        for ii in range(round(0.1 * len(interactions))):
            item = int(random.uniform(0, len(interactions)))
            while item_interaction[interactions[item]] <= cold_thre:
                item = int(random.uniform(0, len(interactions)))
            item_interaction[interactions[item]] -= 1
            validation_data[i].append(interactions[item])
            interactions.pop(item)
            item = int(random.uniform(0, len(interactions)))
            while item_interaction[interactions[item]] <= cold_thre:
                item = int(random.uniform(0, len(interactions)))
            item_interaction[interactions[item]] -= 1
            test_data[i].append(interactions[item])
            interactions.pop(item)
    #print(item_interaction)
    return user_interaction, validation_data, test_data

def dataset_split_sparse(interaction):
    rd.shuffle(interaction)
    n = int(len(interaction) * 0.1)
    test_interaction = interaction[0: n]
    validation_interaction = interaction[n: 2*n]
    train_interaction = interaction[2*n: -1]
    user_num = 0
    for [user, item ] in interaction:
        user_num = max(user_num, user)
    user_num += 1
    train_data = [[] for x in range(user_num)]
    test_data = [[] for x in range(user_num)]
    validation_data = [[] for x in range(user_num)]
    for [user, item] in train_interaction:
        train_data[user].append(item)
    for [user, item] in test_interaction:
        test_data[user].append(item)
    for [user, item] in validation_interaction:
        validation_data[user].append(item)
    return train_data, validation_data, test_data

if __name__ == '__main__':
    dataset_index = 1       # 0:movie, 1:video, 2:cd, 3:clothes
    core = 1                # filter the dataset with x-core
    cold_thre = 0           # the threshold of cold user/item (with less than cold_thre interactions) in training set
    reserve_pro = 0.5       # reserve a part of the dataset to create sparse dataset
    sparse_dense = 1        # 0 for dense split, 1 for sparse split
    print('dataset_index: ', dataset_index)
    print('core: ', core)
    print('cold_thre: ', cold_thre)
    print('reserve_pro: ', reserve_pro)
    print('sparse_dense: ', sparse_dense)

    dataset = ['movie', 'video', 'cd', 'clothes'][dataset_index]
    dataset_name = {'movie': 'Movies_and_TV_5',
                    'video': 'Amazon_Instant_Video_5',
                    'video1': 'Amazon_Instant_Video_5',
                    'video2': 'Amazon_Instant_Video_5',
                    'video3': 'Amazon_Instant_Video_5',
                    'cd': 'CDs_and_Vinyl_5',
                    'cd1': 'CDs_and_Vinyl_5',
                    'cd2': 'CDs_and_Vinyl_5',
                    'cd3': 'CDs_and_Vinyl_5',
                    'music': 'Digital_Music_5',
                    'book': 'Books_5',
                    'clothes': 'Clothing_Shoes_and_Jewelry_5'}[dataset]
    path_read = '../dataset/' + dataset + '/'+dataset_name+'.json'
    path_train = '../dataset/' + dataset + '/train_data.json'
    path_test = '../dataset/' + dataset + '/test_data.json'
    path_validation = '../dataset/' + dataset + '/validation_data.json'
    path_word2vec = '../dataset/google.bin'
    path_user_text = '../dataset/' + dataset + '/user_text.json'
    path_item_text = '../dataset/' + dataset + '/item_text.json'
    path_text = '../dataset/' + dataset + '/text.json'
    print('reading data ...')
    interaction = read_data(path_read)
    rd.shuffle(interaction)
    interaction = interaction[0: int((len(interaction) - 1) * reserve_pro)]   # split out a sparse subset
    print('filtering data ...')
    interaction = dataset_filtering(interaction, core)  # to remove the users and items less than core interactions
    print('encoding data ...')
    interaction, user_text, item_text, word2num = index_encoding(interaction)
    print('loading features ...')
    semantic_matrix = semantic_embedding(word2num, path_word2vec)
    print('splitting dataset ...')
    if sparse_dense == 0: train_data, validation_data, test_data = dataset_split_dense(interaction)
    else: train_data, validation_data, test_data = dataset_split_sparse(interaction)
    print('saving data ...')
    write_data(path_train, train_data)
    write_data(path_validation, validation_data)
    write_data(path_test, test_data)
    write_data(path_user_text, user_text)
    write_data(path_item_text, item_text)
    write_data(path_text, semantic_matrix.tolist())
