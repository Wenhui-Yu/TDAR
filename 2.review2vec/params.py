## Hyper-parameter setting for our proposed models TMN and TCF (i.e., TMF)
## Wenhui Yu 2020.06.02
## author @Wenhui Yu, yuwh16@mails.tsinghua.edu.cn

model = 2           # 0:MF, 1:TMN, 2:TCF (TMF)
dataset = 1         # 0:movie, 1:video, 2:cd, 3:clothes
validate_test = 0   # 0:Validate, 1: Test

DATASET = ['movie', 'video', 'cd', 'clothes'][dataset]
MODEL = ['MF', 'TMN', 'TMF'][model]

LR = [[0.005,0.0001,0.01], [0.05,0.00002,0.02],
      [0.05,0.00002,0.002],[0.1,0.00005,0.05]][dataset][model]

LAMDA = [[0.2,0.02,0.01], [0.1,0.2,0.2],
         [5,2,20],[0.002,0.2,0.1]][dataset][model]

EMB_DIM = 128
BATCH_SIZE = 1024
SAMPLE_RATE = 1
N_EPOCH = 200
TEST_VALIDATION = ['Validation', 'Test'][validate_test]
TOP_K = [2, 5, 10, 20, 50, 100]
IF_SAVE_EMB = 0   # 1: save, otherwise: not save
GPU_INDEX = "0"

DIR = '../dataset/'+DATASET+'/'