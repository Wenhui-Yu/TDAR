## Hyper-parameters
## Wenhui Yu 2020.06.02
## author @Wenhui Yu, yuwh16@mails.tsinghua.edu.cn

dataset_T = 0                        # 0:video, 1:cd, 2:clothes
validate_test = 0                    # 0:Validate, 1: Test

DATASET_T = ['video', 'cd', 'clothes'][dataset_T]
DATASET_S = 'movie'
MODEL = 'TDAR'
OPTIMIZATION = 'Adam'
LR_REC = [0.2, 1, 0.0005][dataset_T]
LAMDA = [0.2,0.2,5][dataset_T]
LR_REC_s = 0.01
LAMDA_s = 0.01
LR_DOM_pos = 0.01
LR_DOM_neg = 0.0001
LAYER = 4
EMB_DIM = 128
BATCH_SIZE = 1024
SAMPLE_RATE = [0, 1]
IF_PRETRAIN = 0
N_EPOCH = 200
TEST_VALIDATION = ['Validation', 'Test'][validate_test]
TOP_K = [2, 5, 10, 20, 50, 100]
gpu_index = "0"
 