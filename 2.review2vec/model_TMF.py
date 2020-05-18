# text-aware MF, the text features are extracted by TMN model.

import tensorflow as tf
import numpy as np

class model_TMF(object):
    def __init__(self, n_users, n_items, emb_dim, lr, lamda, review_embeddings):
        self.model_name = 'TMF'
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.lr = lr
        self.lamda = lamda
        [self.P, self.Q] = review_embeddings

        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.items = tf.placeholder(tf.int32, shape=(None,))
        self.label = tf.placeholder(tf.float32, shape=(None,))


        self.user_embeddings = tf.Variable(tf.random_normal([self.n_users, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='user_factors')
        self.item_embeddings = tf.Variable(tf.random_normal([self.n_items, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='item_factors')
        
        self.user_review = tf.constant(self.P, name='user_review')
        self.item_review = tf.constant(self.Q, name='item_review')
        self.text_score = tf.matmul(self.user_review, self.item_review, transpose_a=False, transpose_b=True)
        
        self.u_embeddings = tf.nn.embedding_lookup(self.user_embeddings, self.users)
        self.i_embeddings = tf.nn.embedding_lookup(self.item_embeddings, self.items)
        self.ui_text_score = tf.gather_nd(self.text_score, tf.transpose(tf.concat([[self.users], [self.items]], axis=0)))
        
        self.all_ratings = tf.matmul(self.user_embeddings, self.item_embeddings, transpose_a=False, transpose_b=True) + self.text_score
        self.loss = self.cross_entropy_loss(self.u_embeddings, self.i_embeddings, self.label) + \
                    self.lamda * self.regularization(self.u_embeddings, self.i_embeddings)
   
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.updates = self.opt.minimize(self.loss)

    def cross_entropy_loss(self, u_emb, i_emb, label):
        epsilon = 0.1 ** 10
        scores = tf.sigmoid(tf.reduce_sum(tf.multiply(u_emb, i_emb), axis=1) + self.ui_text_score)
        maxi = tf.multiply(label, tf.log(scores+epsilon)) + tf.multiply((1-label), tf.log(1-scores+epsilon))
        return tf.negative(tf.reduce_sum(maxi))

    def regularization(self, u_emb, i_emb):
        regularizer = tf.nn.l2_loss(u_emb) + tf.nn.l2_loss(i_emb)
        return regularizer
    