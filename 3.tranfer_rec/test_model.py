## test the model and return the performance

from evaluation import *
from read_data import *
from params import DATASET_T
from params import DATASET_S
from params import TOP_K
from params import TEST_VALIDATION
import operator
import multiprocessing
cores = multiprocessing.cpu_count()

def test_one_user(x):
    k_num = len(TOP_K)
    f1 = np.zeros(k_num)
    ndcg = np.zeros(k_num)
    [score, train_data, test_data] = x
    score_min = - 10 ** 5
    for item in train_data:
        score[item] = score_min
    order = list(np.argsort(score))
    order.reverse()
#     for item in train_data:
#         order.remove(item)
    for i in range(k_num):
        f1[i] += evaluation_F1(order, TOP_K[i], test_data)
        ndcg[i] += evaluation_NDCG(order, TOP_K[i], test_data)
    return f1, ndcg

def test_model(sess, model, label):
    if label == 't':
        user_ratings = sess.run(model.all_ratings_t)
        DIR = '../dataset/' + DATASET_T + '/'
    else:
        user_ratings = sess.run(model.all_ratings_s)
        DIR = '../dataset/' + DATASET_S + '/'
    train_path = DIR + 'train_data.json'
    teat_path = DIR + 'test_data.json'
    validation_path = DIR + 'validation_data.json'
    ## load data
    train_data = read_data(train_path, 1)[0]
    teat_vali_path = validation_path if operator.eq(TEST_VALIDATION, 'Validation') == 1 else teat_path
    test_data = read_data(teat_vali_path, 1)[0]
    result = []
    for u in range(len(user_ratings)):
        if len(test_data[u]) > 0:
            result.append(test_one_user([user_ratings[u], train_data[u], test_data[u]]))
    result = np.array(result)
    F1, NDCG = np.mean(np.array(result), axis=0)
    return F1, NDCG

# def test_model(sess, model, label):
#     if label == 't':
#         User_ratings = sess.run(model.all_ratings_t)
#         DIR = '../dataset/' + DATASET_T + '/'
#     else:
#         User_ratings = sess.run(model.all_ratings_s)
#         DIR = '../dataset/' + DATASET_S + '/'
#     train_path = DIR + 'train_data.json'
#     teat_path = DIR + 'test_data.json'
#     validation_path = DIR + 'validation_data.json'
#     ## load data
#     Train_data = read_data(train_path, 1)[0]
#     teat_vali_path = validation_path if operator.eq(TEST_VALIDATION, 'Validation') == 1 else teat_path
#     Test_data = read_data(teat_vali_path, 1)[0]
#     train_data = []
#     test_data = []
#     user_ratings = []
#     for u in range(np.shape(User_ratings)[0]):
#         if len(Test_data[u]) > 0:
#             train_data.append(Train_data[u])
#             test_data.append(Test_data[u])
#             user_ratings.append(User_ratings[u])
#     user_ratings = np.array(user_ratings).astype(np.float32)
#     pool = multiprocessing.Pool(cores)
#     user_id_rating = zip(user_ratings, train_data, test_data)
#     result = pool.map(test_one_user, user_id_rating)
#     pool.close()
#     F1, NDCG = np.mean(np.array(result), axis=0)
#     return F1, NDCG

def test_domain(sess, model):
    score_u = sess.run(model.domain_disc_u)
    score_i = sess.run(model.domain_disc_i)
    num = 0
    for score in score_u[0]:
        if score > 0:
            num += 1
    for score in score_u[1]:
        if score < 0:
            num += 1
        prec_u = num * 100.0 / (len(score_u[0]) + len(score_u[1]))
    num = 0
    for score in score_i[0]:
        if score > 0:
            num += 1
    for score in score_i[1]:
        if score < 0:
            num += 1
        prec_i = num * 100.0 / (len(score_i[0]) + len(score_i[1]))
    return prec_u, prec_i

