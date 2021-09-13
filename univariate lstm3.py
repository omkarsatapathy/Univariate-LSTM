import pandas as pd 
import numpy as np 
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf 
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten

x = np.linspace(1, 30, 3000)
sine = np.linspace(1, 30, 3000)
scaler = MinMaxScaler()
random = np.random.rand(3000)
for i in range(0, 3000):
  sine[i] = 0.5*(x[i]**(2))+ x[i]*math.sin(x[i])
#sine_scaled = (sine - min(sine)) / (max(sine) - min(sine))
plt.figure(figsize=(15,10))
plt.plot(sine, 'black')
plt.show()

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(sine).reshape(-1,1))
print("df1.shape is  ", df1.shape)



def prepare_data(timeseries_data, n_features):
	X, y =[],[]
	for i in range(len(timeseries_data)):
		# find the end of this pattern
		end_ix = i + n_features
		# check if we are beyond the sequence
		if end_ix > len(timeseries_data)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = timeseries_data[i:end_ix], timeseries_data[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)


timeseries_data = df1
n_steps = 150
X, y = prepare_data(timeseries_data, n_steps)
pred_set = np.linspace(0, n_steps+1, n_steps+1)
for i in range(0,n_steps+1):
  j = len(timeseries_data)-1 - i
  k = len(pred_set)-1 - i 
  pred_set [k] = timeseries_data[j]


print("X shape and y shape is\n")
print(X.shape),print(y.shape)


n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
X.shape


model = keras.Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(80, activation='relu', return_sequences=True,))
model.add(LSTM(70, activation='relu', return_sequences=True,))
model.add(LSTM(50, activation='relu', return_sequences=True,))
model.add(LSTM(30, activation='relu', ))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mae')

history = model.fit(X, y, epochs=50, verbose=1)

x_input = pred_set
temp_input=list(x_input)
lst_output=[]
i=0
while(i<1000):
    
    if(len(temp_input)>3):
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        #print(x_input)
        x_input = x_input.reshape((1, n_steps, n_features))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.append(yhat[0][0])
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.append(yhat[0][0])
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        #print(yhat[0])
        temp_input.append(yhat[0][0])
        lst_output.append(yhat[0][0])
        i=i+1
    

print('lst output is  ', lst_output)

day_new=np.arange(0, 3000)
day_pred=np.arange(3000,4000)

plt.plot(day_new,timeseries_data)
#plt.plot(lst_output)
plt.plot(day_pred,lst_output)
