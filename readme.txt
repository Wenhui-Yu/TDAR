Codes for paper:
Wenhui Yu, Xiao Lin, Junfeng Ge, Wenwu Ou, and Zheng Qin. 2020. Semisupervised Collaborative Filtering by Text-enhanced Domain Adaptation. In Proceedings of the 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ¡¯20), August 23¨C27, 2020, Virtual Event, CA, USA. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3394486.3403264

This project is for our model TMN, TCF, and TDAR.

* Environment:
  Python 3.6.8 :: Anaconda, Inc.
* Libraries:
  tensorflow 1.12.0
  numpy 1.16.4
  pandas 0.18.1
  openpyxl 2.3.2
  xlrd 1.0.0
  xlutils 2.0.0

Please follow the steps below:
1. Download datasets and text features:
   https://drive.google.com/open?id=1Kk2S3JtEf9LHKpMPbrXL2KxbnVzl6f0f
   https://pan.baidu.com/s/1kd_TBLrR1i_1BCjvnj1U4w (password:bvbp)
You can choose one of these two URLs for downloading. Download and unzip dataset1.zip and use it to replace the folder dataset in our project.

2. MF, TMN, and TCF are in folder 2.review2vec.
   Running file _main.py in 2.review2vec.
   If you want to save the embeddings, set IF_SAVE_EMB in params.py as 1 (set _SAVE_EMB as 1 only if you are sure to save the embeddings, or the previous embeddings will be overwritten).
3. TDAR is in folder 3.tranfer_rec.
   Running file _main.py in 3.tranfer_rec.
4. Check the result in folder experiment_result.


