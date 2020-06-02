## Text-enhanced Domain Adaptation for Recommendation (TDAR)
## Wenhui Yu 2020.06.02
## author @Wenhui Yu, yuwh16@mails.tsinghua.edu.cn

import tensorflow as tf
import numpy as np

class model_TDAR(object):
    def __init__(self, layer, n_users_t, n_items_t, n_users_s, n_items_s, 
                 emb_dim, lr_rec, lr_dom_pos, lr_dom_neg, lamda, lr_rec_s,
                 lamda_s, optimization, pretrain_t, pretrain_s, 
                 review_embeddings_t, review_embeddings_s, if_pretrain):
        self.model_name = 'TDAR'
        self.n_users_t = n_users_t
        self.n_items_t = n_items_t
        self.n_users_s = n_users_s
        self.n_items_s = n_items_s
        self.emb_dim = emb_dim
        self.lr_rec_t = lr_rec
        self.lr_rec_s = lr_rec_s
        self.lamda_t = lamda
        self.lamda_s = lamda_s
        self.lr_dom_pos = lr_dom_pos
        self.lr_dom_neg = lr_dom_neg

        self.layer = layer
        self.optimization = optimization
        [self.U_t, self.V_t] = pretrain_t
        [self.U_s, self.V_s] = pretrain_s
        [self.P_t, self.Q_t] = review_embeddings_t
        [self.P_s, self.Q_s] = review_embeddings_s
        self.if_pretrain = if_pretrain
        self.layer_size = [self.emb_dim + np.shape(self.P_t)[1]]
        for l in range(self.layer):
            self.layer_size.append(64)

        self.users_t = tf.placeholder(tf.int32, shape=(None,))
        self.items_t = tf.placeholder(tf.int32, shape=(None,))
        self.rec_label_t = tf.placeholder(tf.float32, shape=(None,))
        self.domain_label_t = tf.placeholder(tf.float32, shape=(None,))
        self.users_s = tf.placeholder(tf.int32, shape=(None,))
        self.items_s = tf.placeholder(tf.int32, shape=(None,))
        self.rec_label_s = tf.placeholder(tf.float32, shape=(None,))
        self.domain_label_s = tf.placeholder(tf.float32, shape=(None,))

        ## initialization:
        if self.if_pretrain:
            self.user_factors_t = tf.Variable(self.U_t)
            self.item_factors_t = tf.Variable(self.V_t)
        else:
            self.user_factors_t = tf.Variable(tf.random_normal([self.n_users_t, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32))
            self.item_factors_t = tf.Variable(tf.random_normal([self.n_items_t, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32))
        self.user_factors_s = tf.Variable(self.U_s[:,0: emb_dim])
        self.item_factors_s = tf.Variable(self.V_s[:,0: emb_dim])

        self.user_review_t = tf.constant(self.P_t)
        self.item_review_t = tf.constant(self.Q_t)
        self.user_review_s = tf.constant(self.P_s)
        self.item_review_s = tf.constant(self.Q_s)

#         self.user_embeddings_t = tf.concat([self.normalize(self.user_factors_t), self.normalize(self.user_review_t)], axis=1)
#         self.item_embeddings_t = tf.concat([self.normalize(self.item_factors_t), self.normalize(self.item_review_t)], axis=1)
#         self.user_embeddings_s = tf.concat([self.normalize(self.user_factors_s), self.normalize(self.user_review_s)], axis=1)
#         self.item_embeddings_s = tf.concat([self.normalize(self.item_factors_s), self.normalize(self.item_review_s)], axis=1)
        self.user_embeddings_t = tf.concat([self.user_factors_t, self.user_review_t], axis=1)
        self.item_embeddings_t = tf.concat([self.item_factors_t, self.item_review_t], axis=1)
        self.user_embeddings_s = tf.concat([self.user_factors_s, self.user_review_s], axis=1)
        self.item_embeddings_s = tf.concat([self.item_factors_s, self.item_review_s], axis=1)

        self.text_score_t = tf.matmul(self.user_review_t, self.item_review_t, transpose_a=False, transpose_b=True)
        self.text_score_s = tf.matmul(self.user_review_s, self.item_review_s, transpose_a=False, transpose_b=True)

        ## domain discriminator
        self.W_u = []
        self.W_i = []
        self.b_u = []
        self.b_i = []
        for l in range(self.layer):
            self.W_u.append(tf.Variable(tf.random_normal([self.layer_size[l], self.layer_size[l + 1]], mean=0.01, stddev=0.02, dtype=tf.float32)))
            self.W_i.append(tf.Variable(tf.random_normal([self.layer_size[l], self.layer_size[l + 1]], mean=0.01, stddev=0.02, dtype=tf.float32)))
            self.b_u.append(tf.Variable(tf.random_normal([1, self.layer_size[l + 1]], mean=0.01, stddev=0.02, dtype=tf.float32)))
            self.b_i.append(tf.Variable(tf.random_normal([1, self.layer_size[l + 1]], mean=0.01, stddev=0.02, dtype=tf.float32)))
        self.h_u = tf.Variable(tf.random_normal([1, self.layer_size[-1]], mean=0.01, stddev=0.02, dtype=tf.float32))
        self.h_i = tf.Variable(tf.random_normal([1, self.layer_size[-1]], mean=0.01, stddev=0.02, dtype=tf.float32))
        self.d_u = tf.Variable(-0.1)
        self.d_i = tf.Variable(-0.1)

        self.u_factors_t = tf.nn.embedding_lookup(self.user_factors_t, self.users_t)
        self.i_factors_t = tf.nn.embedding_lookup(self.item_factors_t, self.items_t)
        self.u_factors_s = tf.nn.embedding_lookup(self.user_factors_s, self.users_s)
        self.i_factors_s = tf.nn.embedding_lookup(self.item_factors_s, self.items_s)
        
        self.u_embeddings_t = tf.nn.embedding_lookup(self.user_embeddings_t, self.users_t)
        self.i_embeddings_t = tf.nn.embedding_lookup(self.item_embeddings_t, self.items_t)
        self.u_embeddings_s = tf.nn.embedding_lookup(self.user_embeddings_s, self.users_s)
        self.i_embeddings_s = tf.nn.embedding_lookup(self.item_embeddings_s, self.items_s)
        
        self.ui_text_score_t = tf.gather_nd(self.text_score_t, tf.transpose(tf.concat([[self.users_t], [self.items_t]], axis=0)))
        self.ui_text_score_s = tf.gather_nd(self.text_score_s, tf.transpose(tf.concat([[self.users_s], [self.items_s]], axis=0)))
        
        ## for model testing
        self.all_ratings_t = tf.matmul(self.user_factors_t, self.item_factors_t, transpose_a=False, transpose_b=True) + self.text_score_t
        self.all_ratings_s = tf.matmul(self.user_factors_s, self.item_factors_s, transpose_a=False, transpose_b=True) + self.text_score_s
        self.domain_disc_u = self.domain_rating(self.user_embeddings_t, self.user_embeddings_s, self.W_u, self.b_u, self.h_u, self.d_u)
        self.domain_disc_i = self.domain_rating(self.item_embeddings_t, self.item_embeddings_s, self.W_i, self.b_i, self.h_i, self.d_i)

        self.loss_rec_t = self.cross_entropy_loss(self.inter(self.u_factors_t, self.i_factors_t), self.ui_text_score_t, self.rec_label_t) + self.lamda_t*self.regularization([self.u_factors_t, self.i_factors_t])
        self.loss_rec_s = self.cross_entropy_loss(self.inter(self.u_factors_s, self.i_factors_s), self.ui_text_score_s, self.rec_label_s) + self.lamda_s*self.regularization([self.u_factors_s, self.i_factors_s])

        self.loss_domain_u = self.cross_entropy_loss(self.mlp(self.u_embeddings_t, self.W_u, self.b_u, self.h_u, self.d_u), 0, self.domain_label_t) + \
                             self.cross_entropy_loss(self.mlp(self.u_embeddings_s, self.W_u, self.b_u, self.h_u, self.d_u), 0, self.domain_label_s)

        self.loss_domain_i = self.cross_entropy_loss(self.mlp(self.i_embeddings_t, self.W_i, self.b_i, self.h_i, self.d_i), 0, self.domain_label_t) + \
                             self.cross_entropy_loss(self.mlp(self.i_embeddings_s, self.W_i, self.b_i, self.h_i, self.d_i), 0, self.domain_label_s)
        
        self.opt1_t = tf.train.GradientDescentOptimizer(learning_rate=self.lr_rec_t)
        self.opt1_s = tf.train.GradientDescentOptimizer(learning_rate=self.lr_rec_s)
        self.opt2_u = tf.train.AdamOptimizer(learning_rate=self.lr_dom_pos)
        self.opt2_i = tf.train.AdamOptimizer(learning_rate=self.lr_dom_pos)
        self.opt3_u = tf.train.AdamOptimizer(learning_rate=self.lr_dom_neg)
        self.opt3_i = tf.train.AdamOptimizer(learning_rate=self.lr_dom_neg)

        self.update1_t = self.opt1_t.minimize(self.loss_rec_t, var_list=[self.user_factors_t, self.item_factors_t])
        self.update1_s = self.opt1_s.minimize(self.loss_rec_s, var_list=[self.user_factors_s, self.item_factors_s])
        self.update2_u = self.opt2_u.minimize(self.loss_domain_u, var_list=[self.h_u, self.d_u] + self.W_u + self.b_u)
        self.update2_i = self.opt2_i.minimize(self.loss_domain_i, var_list=[self.h_i, self.d_i] + self.W_i + self.b_i)
        self.update3_u = self.opt3_u.minimize(-self.loss_domain_u, var_list=[self.user_factors_t, self.user_factors_s])
        self.update3_i = self.opt3_i.minimize(-self.loss_domain_i, var_list=[self.item_factors_t, self.item_factors_s])

    def cross_entropy_loss(self, ui_inter_score, ui_text_score, label):
        score = tf.nn.sigmoid(ui_inter_score + ui_text_score)
        epsilon = 0.1 ** 5
        maxi = tf.multiply(label, tf.log(score + epsilon)) + tf.multiply((1 - label), tf.log(1 - score + epsilon))
        return tf.negative(tf.reduce_sum(maxi))

    def regularization(self, Para):
        regularizer = 0
        for para in Para:
            regularizer += tf.nn.l2_loss(para)
        return regularizer

    def inter(self, u_emb, i_emb):
        return tf.reduce_sum(tf.multiply(u_emb, i_emb), axis=1)

    def mlp(self, emb, W, b, h, d):
        for l in range(self.layer):
            # emb = tf.nn.sigmoid(tf.matmul(emb, W[l]) + b[l])
            emb = tf.nn.relu(tf.matmul(emb, W[l]) + b[l])
        score = tf.matmul(emb, h, transpose_a=False, transpose_b=True)
        return tf.reshape(score, [-1]) + d

    def domain_rating(self, emb_t, emb_s, W, b, h, d):
        score = []
        score.append(self.mlp(emb_t, W, b, h, d))
        score.append(self.mlp(emb_s, W, b, h, d))
        return score

    def normalize(self, features):
        features_norm = tf.norm(features, axis=1)
        features_norm = tf.reduce_mean(features_norm)
        return features/features_norm

