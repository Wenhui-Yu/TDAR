## MF
## Optimized by point-wise cross entropy loss
## Wenhui Yu 2020.06.02
## author @Wenhui Yu, yuwh16@mails.tsinghua.edu.cn

import tensorflow as tf

class model_MF(object):
    def __init__(self,n_users,n_items,emb_dim,lr,lamda):
        self.model_name = 'MF'
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.lr = lr
        self.lamda = lamda

        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.items = tf.placeholder(tf.int32, shape=(None,))
        self.label = tf.placeholder(tf.float32, shape=(None,))

        self.user_embeddings = tf.Variable(
            tf.random_normal([self.n_users, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
            name='user_embeddings')
        self.item_embeddings = tf.Variable(
            tf.random_normal([self.n_items, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
            name='item_embeddings')

        self.u_embeddings = tf.nn.embedding_lookup(self.user_embeddings, self.users)
        self.i_embeddings = tf.nn.embedding_lookup(self.item_embeddings, self.items)

        self.all_ratings = tf.matmul(self.u_embeddings, self.i_embeddings, transpose_a = False, transpose_b = True)

        self.loss = self.cross_entropy_loss(self.u_embeddings, self.i_embeddings, self.label)

        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.updates = self.opt.minimize(self.loss, var_list=[self.user_embeddings, self.item_embeddings])

    def cross_entropy_loss(self, users, items, label):
        epsilon = 0.1 ** 10
        scores = tf.sigmoid(tf.reduce_sum(tf.multiply(users, items), axis=1))
        maxi = tf.multiply(label, tf.log(scores+epsilon)) + tf.multiply((1-label), tf.log(1-scores+epsilon))
        return tf.negative(tf.reduce_sum(maxi))
