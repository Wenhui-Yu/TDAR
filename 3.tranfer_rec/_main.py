## author@Wenhui Yu  email:yuwh16@tsinghua.edu.cn  2019.04.14
## Transfer learning recommendation (TL) and baselines

from train_model import *
from params import *
from print_save import *
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
print('GPU_INDEX:  ', gpu_index)

if __name__ == '__main__':
    path_excel = '../experiment_result/'+DATASET_T+'_'+MODEL+'_'+str(int(time.time()))+str(int(random.uniform(100,900)))+'.xlsx'
    para = [DATASET_T,DATASET_S,MODEL,LR_REC,LR_DOM_pos,LR_DOM_neg,LAMDA,
            LR_REC_s,LAMDA_s,LAYER,EMB_DIM,BATCH_SIZE,SAMPLE_RATE,N_EPOCH,
            TEST_VALIDATION,TOP_K,OPTIMIZATION,IF_PRETRAIN]
    para_name = ['DATASET','DATASET_SOURCE','MODEL','LR_REC','LR_DOM_pos','LR_DOM_neg','LAMDA',
                 'LR_REC_s','LAMDA_s','LAYER','EMB_DIM','BATCH_SIZE','SAMPLE_RATE','N_EPOCH',
                 'TEST_VALIDATION','TOP_K','OPTIMIZATION','IF_PRETRAIN']
    ## print and save model hyperparameters
    print_params(para_name, para)
    save_params(para_name, para, path_excel)
    ## train the model
    train_model(para, path_excel)

    # try: train_model(para, path_excel)
    # except: continue
