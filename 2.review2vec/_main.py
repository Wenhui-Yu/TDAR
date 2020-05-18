## For TMN and TCF

from train_model import *
from params import *
from print_save import *
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX
print('GPU INDEX: ', GPU_INDEX)

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    path_excel = '../experiment_result/' + DATASET + '_' + MODEL + '_' + str(int(time.time())) + str(int(random.uniform(100, 900))) + '.xlsx'
    para = [DATASET,MODEL,LR,LAMDA,EMB_DIM, BATCH_SIZE,SAMPLE_RATE,N_EPOCH,TEST_VALIDATION,TOP_K]
    para_name = ['DATASET','MODEL','LR','LAMDA','EMB_DIM','BATCH_SIZE','SAMPLE_RATE','N_EPOCH','TEST_VALIDATION','TOP_K']
    ## print and save model hyperparameters
    print_params(para_name, para)
    save_params(para_name, para, path_excel)
    ## train the model
    train_model(para, path_excel, IF_SAVE_EMB)
