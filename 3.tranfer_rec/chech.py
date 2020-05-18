
import numpy as np
import tensorflow as tf
import random as rd
a = np.array([[1,2,3],[2,3,4],[1,3,4]]).astype(np.float32)
f_u=tf.Variable(a)
f_u_norm1 = tf.norm(f_u, axis=1)#  np.linalg.norm(features, axis=1, keepdims=True)
f_u_norm2 = tf.reduce_mean(f_u_norm1)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(f_u_norm1))
# f_v = f_u
#
# f_u_norm = np.linalg.norm(f_u, axis=1, keepdims=True)
# print(f_u_norm)
# f_v_norm = np.linalg.norm(f_v, axis=1, keepdims=True)
# f_u_norm = np.mean(f_u_norm)
# print(f_u_norm)
# f_v_norm = np.mean(f_v_norm)
# print(f_u/f_u_norm)
