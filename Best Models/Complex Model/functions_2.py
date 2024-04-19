import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,SimpleRNN,Dense,Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from datetime import datetime
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from pandas import DataFrame
from pandas import concat
import keras

keras.utils.set_random_seed(1234)


#################

def missing_value_handler(dataset):
     interpolated_dataset = dataset.interpolate()
     interpolated_dataset = interpolated_dataset.apply(pd.to_numeric, errors='coerce')

     return(interpolated_dataset)

#########################################

def dataset_scaling(dataset):

    values = dataset.values
    values = values.astype('float32')
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    return(scaled)

##################################

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
 """
 Frame a time series as a supervised learning dataset.
 Arguments:
 data: Sequence of observations as a list or NumPy array.
 n_in: Number of lag observations as input (X).
 n_out: Number of observations as output (y).
 dropnan: Boolean whether or not to drop rows with NaN values.
 Returns:
 Pandas DataFrame of series framed for supervised learning.
 """
 n_vars = 1 if type(data) is list else data.shape[1]
 df = DataFrame(data)
 cols, names = list(), list()
 # input sequence (t-n, ... t-1)
 for i in range(n_in, 0, -1):
  cols.append(df.shift(i))
  names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
  # forecast sequence (t, t+1, ... t+n)
 for i in range(0, n_out):
  cols.append(df.shift(-i))
  if i == 0:
   names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
  else:
   names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
 # put it all together
 agg = concat(cols, axis=1)
 agg.columns = names
 # drop rows with NaN values
 if dropnan:
  agg.dropna(inplace=True)
 return agg




##########################################

def rnn_trainer(dataset,df,n,rnn_units,dense_units,air_property,name,name1,epochs = 100,batch_size = 32,learning_rate = 0.001):


   dict = {"AQHI":-1,"PM2.5":-16,"PM10":-15,"SO2":-14,"NO2":-13,"CO":-12,"O3":-11}

   # split into train sets、validation sets and test sets
   values = dataset.values
   number=len(values)
   n_train_hours = int(number*0.7)
   n_valid_hours = int(number*0.9)
   train = values[:n_train_hours, :]
   valid = values[n_train_hours:n_valid_hours, :]
   test = values[n_valid_hours:, :]
   # split into input and outputs
   u = 16*n
   train_X, train_y = train[:, :-u], train[:, dict[air_property]]
   valid_X, valid_y = valid[:, :-u], valid[:, dict[air_property]]
   test_X, test_y = test[:, :-u], test[:, dict[air_property]]
   # reshape input to be 3D [samples, timesteps, features]
   train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
   valid_X = valid_X.reshape((valid_X.shape[0], 1, valid_X.shape[1]))
   test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))





   timesteps = train_X.shape[1]
   input_dim = train_X.shape[2]
   output_dim=1



   inputs = keras.Input(shape=(timesteps, input_dim))

   tf.random.set_seed(1234)

   rnn_out = SimpleRNN(rnn_units, activation='relu', return_sequences=False)(inputs)


   dense = Dense(units=dense_units, activation='relu')(rnn_out)

   output = Dense(units=output_dim, activation='linear')(dense)


   rnn_1 = keras.Model(inputs=inputs,  outputs=output)


   rnn_1.compile(loss='mse', optimizer=keras.optimizers.Adam(
       learning_rate=learning_rate), metrics=['mae'])

   callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,restore_best_weights=True)


   history = rnn_1.fit(train_X, train_y, epochs=epochs, batch_size=batch_size,callbacks=[callback], validation_data=(valid_X, valid_y)

                    )
   visualize_loss(history, "Training and Validation Loss",name1)

   errors = evaluate(rnn_1,df,test_X,test_y,air_property,name)

   return(rnn_1,errors)



############################################################



def dataset_pipeline_rnn(dataset,used_timesteps,prediction_time,rnn_units,dense_units,air_property,name = 'time_series.png',name1="loss",epochs = 100,batch_size = 32,learning_rate = 0.001):

     interpolated_dataset = missing_value_handler(dataset)
     scaled_dataset = dataset_scaling(interpolated_dataset)
     series_dataset = series_to_supervised(scaled_dataset, used_timesteps, prediction_time)
     rnn_1 = rnn_trainer(series_dataset,interpolated_dataset,prediction_time,rnn_units,dense_units,air_property,name,name1,epochs,batch_size,learning_rate)

     return(rnn_1)




###############################################

def visualize_loss(history, title,name):

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure(dpi=300)
    plt.plot(epochs, loss,"black", label="Training loss")
    plt.plot(epochs,loss, 'o', markersize = 5,color="black")
    plt.plot(epochs, val_loss,color="gray", label="Validation loss")
    plt.plot(epochs,val_loss, 'o', markersize = 5,color="gray")
    plt.title(name)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    #plt.savefig(name)
    plt.show()
    plt.close()

########################################

def compare_visual(data1, data2,air_property,name='time_series.png'):
    fig, axes = plt.subplots(
        nrows=1, ncols=1, figsize=(20, 15), dpi=300, facecolor="w", edgecolor="k"
    )


    ax = data1[air_property].plot(
            ax=axes,
            color='black',
            rot=25,
            label="actual",
            lw = 4
    )
    ax = data2[air_property].plot(
            ax=axes,
            color='gray',
            rot=25,
            label='prediction',
            lw = 4
    )
    ax.legend(fontsize = 28)

    ax.set_xlabel("Hour",fontsize = 28)
    ax.set_ylabel(air_property,fontsize = 28)
    ax.set_title(name,fontsize = 28)
    ax.tick_params(axis='both', labelsize=28)
    plt.tight_layout()
    #plt.savefig(name)
    plt.show()
    plt.close()



######################################################



def evaluate(model,dataset,test_X,test_y,air_property,name):

    dict = {"AQHI":-1,"PM2.5":-16,"PM10":-15,"SO2":-14,"NO2":-13,"CO":-12,"O3":-11}
    yhat = model.predict(test_X)

    inv_yhat=inv_scale(dataset,yhat,air_property)
    inv_y=inv_scale(dataset,test_y,air_property)


    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    mae = mean_absolute_error(inv_y, inv_yhat)
    print('Test MAE: %.3f' % mae)
    R2=r2_score(inv_y, inv_yhat)
    print('Test R2: %.3f' % R2)

    y1 = DataFrame(
         inv_y[-100:], index=dataset.index[-100:], columns=[dataset.columns[dict[air_property]]])
    y2 = DataFrame(
          inv_yhat[-100:], index=dataset.index[-100:], columns=[dataset.columns[dict[air_property]]])
    compare_visual(y1,y2,air_property,name)
    return(rmse,mae,R2)

################################################

def inv_scale(df, y,air_property):

    dict = {"AQHI":-1,"PM2.5":-16,"PM10":-15,"SO2":-14,"NO2":-13,"CO":-12,"O3":-11}
    max = df.max()[dict[air_property]]
    min = df.min()[dict[air_property]]

    yy=y.copy()
    for i in range(len(yy)):
        yy[i] = yy[i]*(max-min)+min
    return yy


##############################################












