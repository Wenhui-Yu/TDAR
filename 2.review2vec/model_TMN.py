## Text Memory Net (TMN)
## Wenhui Yu 2020.06.02
## author @Wenhui Yu, yuwh16@mails.tsinghua.edu.cn

import tensorflow as tf
import numpy as np

class model_TMN(object):
    def __init__(self, n_users, n_items, emb_dim, lr, lamda, text_embeddings, user_word_num, item_word_num):
        self.model_name = 'TMN'
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.lr = lr
        self.lamda = lamda
        self.text_embeddings = text_embeddings
        self.word_num, self.word_dim = np.shape(text_embeddings)
        self.user_word_num = user_word_num
        self.item_word_num = item_word_num

        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.items = tf.placeholder(tf.int32, shape=(None,))
        self.label = tf.placeholder(tf.float32, shape=(None,))
        self.user_word = tf.placeholder(tf.int32, shape=(None, self.user_word_num))
        self.item_word = tf.placeholder(tf.int32, shape=(None, self.item_word_num))

        self.user_embeddings = tf.Variable(
            tf.random_normal([self.n_users, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
            name='user_embeddings')
        self.item_embeddings = tf.Variable(
            tf.random_normal([self.n_items, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
            name='item_embeddings')
        self.word_embeddings = tf.Variable(
            tf.random_normal([self.word_num, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
            name='word_embeddings')


        self.u_embeddings = tf.nn.embedding_lookup(self.user_embeddings, self.users)
        self.i_embeddings = tf.nn.embedding_lookup(self.item_embeddings, self.items)
        self.u_text = tf.nn.embedding_lookup(self.word_embeddings, self.user_word)
        self.i_text = tf.nn.embedding_lookup(self.word_embeddings, self.item_word)
        self.u_word_embeddings = tf.nn.embedding_lookup(self.text_embeddings, self.user_word)
        self.i_word_embeddings = tf.nn.embedding_lookup(self.text_embeddings, self.item_word)

        k = 100
        print(k)
        self.u_text_attention = tf.stack([self.u_embeddings] * self.user_word_num, 1)
        self.u_text_attention = tf.multiply(self.u_text, self.u_text_attention)
        self.u_text_attention = tf.reduce_sum(self.u_text_attention, axis=2)
        self.u_text_attention = tf.nn.softmax(k * self.u_text_attention)
        self.i_text_attention = tf.stack([self.i_embeddings] * self.item_word_num, 1)
        self.i_text_attention = tf.multiply(self.i_text, self.i_text_attention)
        self.i_text_attention = tf.reduce_sum(self.i_text_attention, axis=2)
        self.i_text_attention = tf.nn.softmax(k * self.i_text_attention)

        self.u_text_embeddings = tf.stack([self.u_text_attention] * self.word_dim, 2)
        self.u_text_embeddings = tf.multiply(self.u_text_embeddings, self.u_word_embeddings)
        self.u_text_embeddings = tf.reduce_sum(self.u_text_embeddings, axis=1)
        self.i_text_embeddings = tf.stack([self.i_text_attention] * self.word_dim, 2)
        self.i_text_embeddings = tf.multiply(self.i_text_embeddings, self.i_word_embeddings)
        self.i_text_embeddings = tf.reduce_sum(self.i_text_embeddings, axis=1)


        self.all_ratings = tf.matmul(self.u_text_embeddings, self.i_text_embeddings, transpose_a=False, transpose_b=True)

        self.loss = self.cross_entropy_loss(self.u_text_embeddings, self.i_text_embeddings, self.label) + \
                    self.lamda * self.regularization(self.u_text, self.i_text)
#         self.loss = self.cross_entropy_loss(self.u_text_embeddings, self.i_text_embeddings, self.label) + \
#                     self.lamda * self.regularization(self.u_text, self.i_text, self.u_embeddings, self.i_embeddings)

        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.updates = self.opt.minimize(self.loss)

    def cross_entropy_loss(self, users, items, label):
        epsilon = 0.1 ** 10
        scores = tf.sigmoid(tf.reduce_sum(tf.multiply(users, items), axis=1))
        maxi = tf.multiply(label, tf.log(scores + epsilon)) + tf.multiply((1 - label), tf.log(1 - scores + epsilon))
        return tf.negative(tf.reduce_sum(maxi))

    def regularization(self, users, items):
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(items)
        return regularizer
#     def regularization(self, u_text, i_text, users, items):
#         regularizer = tf.nn.l2_loss(u_text) + tf.nn.l2_loss(i_text) + tf.nn.l2_loss(users) + tf.nn.l2_loss(items)
#         return regularizer
