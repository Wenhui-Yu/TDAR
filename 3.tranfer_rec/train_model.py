## split train data into batches and train the model
## Wenhui Yu 2020.06.02
## author @Wenhui Yu, yuwh16@mails.tsinghua.edu.cn

from model_TDAR import *
from test_model import *
from read_data import *
from print_save import *
import numpy as np

def train_model(para, path_excel):
    [DATASET_T, DATASET_S,MODEL,LR_REC,LR_DOM_pos,LR_DOM_neg,LAMDA,
     LR_REC_s,LAMDA_s,LAYER,EMB_DIM,BATCH_SIZE,SAMPLE_RATE,N_EPOCH,
     _,TOP_K,OPTIMIZATION,IF_PRETRAIN] = para
    
    ## paths of data
    train_path_t = '../dataset/' + DATASET_T + '/train_data.json'
    train_path_s = '../dataset/' + DATASET_S + '/train_data.json'
    pretrain_path_t = '../dataset/' + DATASET_T + '/latent_embeddings.json'
    pretrain_path_s = '../dataset/' + DATASET_S + '/latent_embeddings.json'
    review_path_t = '../dataset/' + DATASET_T + '/review_embeddings.json'
    review_path_s = '../dataset/' + DATASET_S + '/review_embeddings.json'

    ## load train data
    [train_data_t, train_data_interaction_t, user_num_t, item_num_t] = read_data(train_path_t, BATCH_SIZE)
    [train_data_s, train_data_interaction_s, user_num_s, item_num_s] = read_data(train_path_s, BATCH_SIZE)

    pretrain_s = read_bases(pretrain_path_s)
    review_s = read_bases(review_path_s)
    try:
        pretrain_t = read_bases(pretrain_path_t)
    except:
        print('\n There is no pre-trained feature found !! \n')
        pretrain_t = [0, 0]
        IF_PRETRAIN = 0
    review_t = read_bases(review_path_t)

    ## define the model
    model = model_TDAR(layer=LAYER,n_users_t=user_num_t,n_items_t=item_num_t,
                       n_users_s=user_num_s,n_items_s=item_num_s,emb_dim=EMB_DIM,
                       lr_rec=LR_REC,lr_dom_pos=LR_DOM_pos,lr_dom_neg=LR_DOM_neg,
                       lamda=LAMDA,lr_rec_s=LR_REC_s,lamda_s=LAMDA_s,optimization=OPTIMIZATION,
                       pretrain_t=pretrain_t, pretrain_s=pretrain_s,review_embeddings_t=review_t,
                       review_embeddings_s=review_s, if_pretrain=IF_PRETRAIN)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    ## split the training samples into batches
    batches = list(range(0, len(train_data_interaction_t), BATCH_SIZE))
    batches.append(len(train_data_interaction_t))
    ## training iteratively
    F1_max_t = 0
    F1_max_s = 0
    pre_i_max = 0
    pre_u_max = 0
    F1_df = pd.DataFrame(columns=TOP_K)
    NDCG_df = pd.DataFrame(columns=TOP_K)
    loss_rec_s = 0
    loss_rec_t = 0 
    for epoch in range(N_EPOCH):
        rd.shuffle(train_data_interaction_t)
        rd.shuffle(train_data_interaction_s)
        for batch_num in range(len(batches)-1):
            train_batch_data_t = []
            train_batch_data_s = []
            for sample_t in range(batches[batch_num], batches[batch_num+1]):
                (user_t, pos_item_t) = train_data_interaction_t[sample_t]
                sample_s = random.randint(0, len(train_data_interaction_s)-1)
                (user_s, pos_item_s) = train_data_interaction_s[sample_s]
                train_batch_data_t.append([user_t, pos_item_t, 1, 1])
                train_batch_data_s.append([user_s, pos_item_s, 1, 0])
                # add negatives to the target domain
                sample_num = 0
                while sample_num < SAMPLE_RATE[0]:
                    neg_item_t = int(random.uniform(0, item_num_t))
                    if not (neg_item_t in train_data_t[user_t]):
                        sample_num += 1
                        train_batch_data_t.append([user_t, neg_item_t, 0, 1])
                # add negatives to the source domain
                sample_num = 0
                while sample_num < SAMPLE_RATE[1]:
                    neg_item_s = int(random.uniform(0, item_num_s))
                    if not (neg_item_s in train_data_s[user_s]):
                        sample_num += 1
                        train_batch_data_s.append([user_s, neg_item_s, 0, 0])
            train_batch_data_t = np.array(train_batch_data_t)
            train_batch_data_s = np.array(train_batch_data_s)
            try:
                [update1_t, update1_s, update2_u, update2_i, update3_u, update3_i, loss_rec_t, loss_rec_s,
                 loss_domain_u, loss_domain_i] = sess.run(
                    [model.update1_t, model.update1_s, model.update2_u, model.update2_i, model.update3_u,
                     model.update3_i, model.loss_rec_t, model.loss_rec_s, model.loss_domain_u, model.loss_domain_i],
                    feed_dict={model.users_t: train_batch_data_t[:, 0], model.items_t: train_batch_data_t[:, 1],
                               model.rec_label_t: train_batch_data_t[:, 2], model.domain_label_t: train_batch_data_t[:, 3],
                               model.users_s: train_batch_data_s[:, 0], model.items_s: train_batch_data_s[:, 1],
                               model.rec_label_s: train_batch_data_s[:, 2], model.domain_label_s: train_batch_data_s[:, 3]})
            except:
                update, loss_rec = sess.run([model.updates, model.loss],
                                            feed_dict={model.users: train_batch_data_t[:, 0],
                                                       model.items: train_batch_data_t[:, 1],
                                                       model.labels: train_batch_data_t[:, 2]})


        F1_t, NDCG_t = test_model(sess, model, 't')
        F1_max_t = max(F1_max_t, F1_t[0])
        F1_s, NDCG_s = test_model(sess, model, 's')
        F1_max_s = max(F1_max_s, F1_s[0])
        pre_u, pre_i = test_domain(sess, model)
            
        ## print performance
        print_value([epoch + 1, loss_rec_t, loss_rec_s, loss_domain_u, loss_domain_i, F1_max_t, F1_max_s, pre_u, pre_i, F1_t, NDCG_t,  F1_s, NDCG_s])

        ## save performance
        F1_df.loc[epoch + 1] = F1_t
        NDCG_df.loc[epoch + 1] = NDCG_t
        save_value([[F1_df, 'F1'], [NDCG_df, 'NDCG']], path_excel, first_sheet=False)
        if not (loss_rec_s + loss_rec_t)  < 10**10:
            break
