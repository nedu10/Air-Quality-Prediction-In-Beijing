from functions_2 import *

df = pd.read_csv("Beijing_outlier_removed.csv")

RNN = dataset_pipeline_rnn(df,3,1,64,128,'AQHI',name = 'time_series.png',name1="loss",epochs = 100,batch_size = 32,learning_rate = 0.001)

model,errors = RNN