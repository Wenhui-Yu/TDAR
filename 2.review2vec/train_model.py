## split train data into batches and train the model

from model_MF import *
from model_TMN import *
from model_TMF import *
from test_model import *
from read_data import *
from print_save import *
import gc

def assignment(arr, lis):
    n = len(arr)
    rho = int(n / len(lis)) + 1
    Lis = lis * rho
    return np.array(Lis[0: n])
        
def train_model(para, path_excel, if_save_emb):
    [_,MODEL,LR,LAMDA,EMB_DIM,BATCH_SIZE, SAMPLE_RATE,N_EPOCH,_,_,] = para
    ## paths of data
    train_path = DIR + 'train_data.json'
    save_text_embeddings_path = DIR + 'review_embeddings.json'
    save_latant_embeddings_path = DIR + 'latent_embeddings.json'
    text_embeddings_path = DIR + 'text.json'
    user_review_path = DIR + 'user_text.json'
    item_review_path = DIR + 'item_text.json'
    ## load train data
    [train_data, train_data_interaction, user_num, item_num] = read_data(train_path, BATCH_SIZE)
    if MODEL == 'TMN':
        text_matrix = load_features(text_embeddings_path)
        text_matrix = text_matrix.astype(np.float32)
        user_review = load_data(user_review_path)
        item_review = load_data(item_review_path)
        user_word_num = 0
        for review in user_review:
            user_word_num = max(len(review), user_word_num)
        item_word_num = 0
        for review in item_review:
            item_word_num = max(len(review), item_word_num)
        user_word_num = min(user_word_num, 200)
        item_word_num = min(item_word_num, 200)
        user_review_feature = np.ones((user_num, user_word_num))
        item_review_feature = np.ones((item_num, item_word_num))
        for user in range(user_num):
            user_review_feature[user] = assignment(user_review_feature[user], user_review[user])
        for item in range(item_num):
            item_review_feature[item] = assignment(item_review_feature[item], item_review[item])

    ## define the model
    if MODEL == 'MF': model = model_MF(n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA)
    if MODEL == 'TMN': model = model_TMN(n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA, text_embeddings = text_matrix, user_word_num = user_word_num, item_word_num = item_word_num)
    if MODEL == 'TMF': 
        review_embeddings = read_features(save_text_embeddings_path)
        model = model_TMF(n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA, review_embeddings = review_embeddings)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    ## split the training samples into batches
    batches = list(range(0, len(train_data_interaction), BATCH_SIZE))
    batches.append(len(train_data_interaction))

    ## training iteratively
    F1_max = -10
    F1_df = pd.DataFrame(columns=TOP_K)
    NDCG_df = pd.DataFrame(columns=TOP_K)
    for epoch in range(N_EPOCH):
        for batch_num in range(len(batches)-1):
            train_batch_data = []
            if MODEL == 'TMN':
                user_review_batch = np.ones(((1+SAMPLE_RATE)*(batches[batch_num+1]-batches[batch_num]), user_word_num))
                item_review_batch = np.ones(((1+SAMPLE_RATE)*(batches[batch_num+1]-batches[batch_num]), item_word_num))
            num = 0
            for sample in range(batches[batch_num], batches[batch_num+1]):
                user, pos_item = train_data_interaction[sample]
                sample_num = 0
                train_batch_data.append([user, pos_item, 1]) 
                if MODEL == 'TMN':
                    user_review_batch[num] = user_review_feature[user]
                    item_review_batch[num] = item_review_feature[pos_item]
                num += 1
                while sample_num < SAMPLE_RATE:
                    neg_item = int(random.uniform(0, item_num))
                    if not (neg_item in train_data[user]):
                        sample_num += 1
                        train_batch_data.append([user, neg_item, 0])
                        if MODEL == 'TMN':
                            user_review_batch[num] = user_review_feature[user]
                            item_review_batch[num] = item_review_feature[neg_item]
                        num += 1
            train_batch_data = np.array(train_batch_data)
            try:
                _, loss = sess.run([model.updates, model.loss],
                                   feed_dict={model.users: train_batch_data[:, 0],
                                              model.items: train_batch_data[:, 1],
                                              model.label: train_batch_data[:, 2],
                                              model.user_word: user_review_batch,
                                              model.item_word: item_review_batch})
            except:
                _, loss = sess.run([model.updates, model.loss],
                                   feed_dict={model.users: train_batch_data[:, 0],
                                              model.items: train_batch_data[:, 1],
                                              model.label: train_batch_data[:, 2]})
        if MODEL == 'TMN': F1, NDCG = test_model(sess, model, user_review_feature, item_review_feature)
        else: F1, NDCG = test_model(sess, model, 0, 0)
        if F1_max < F1[0]:
            F1_max = F1[0]
            if if_save_emb == 1:
                try:
                    user_text_embedding = np.zeros((user_num, np.shape(text_matrix)[1]))
                    item_text_embedding = np.zeros((item_num, np.shape(text_matrix)[1]))
                    user_batch_list = list(range(0, user_num, 500))
                    user_batch_list.append(user_num)
                    item_batch_list = list(range(0, item_num, 500))
                    item_batch_list.append(item_num)
                    for u in range(len(user_batch_list) - 1):
                        u1, u2 = user_batch_list[u], user_batch_list[u + 1]
                        user_batch = np.array(range(u1, u2))
                        user_review_batch = user_review_feature[u1: u2]
                        u_text_embedding = sess.run([model.u_text_embeddings],
                                                     feed_dict={model.users: user_batch,
                                                                model.user_word: user_review_batch})
                        user_text_embedding[u1: u2] = u_text_embedding[0]
                    for i in range(len(item_batch_list) - 1):
                        i1, i2 = item_batch_list[i], item_batch_list[i + 1]
                        item_batch = np.array(range(i1, i2))
                        item_review_batch = item_review_feature[i1: i2]
                        i_text_embedding = sess.run([model.i_text_embeddings],
                                                     feed_dict={model.items: item_batch,
                                                                model.item_word: item_review_batch})
                        item_text_embedding[i1: i2] = i_text_embedding[0]
                except:
                    user_embedding, item_embedding = sess.run([model.user_embeddings, model.item_embeddings])            
        ## print performance
        print_value([epoch + 1, loss, F1_max, F1, NDCG])
        F1_df.loc[epoch + 1] = F1
        NDCG_df.loc[epoch + 1] = NDCG
        save_value([[F1_df, 'F1'], [NDCG_df, 'NDCG']], path_excel, first_sheet=False)
        if not loss < 10**10:
            break
    if if_save_emb == 1:
        if MODEL == 'TMN':
            save_embeddings([user_text_embedding.tolist(), item_text_embedding.tolist()], save_text_embeddings_path)
        if MODEL == 'TMF':
            save_embeddings([user_embedding.tolist(), item_embedding.tolist()], save_latant_embeddings_path)
        try: del u_text_embedding, i_text_embedding, user_text_embedding, item_text_embedding
        except: del user_embedding, item_embedding
    del model, loss, _, sess
    gc.collect()
    
