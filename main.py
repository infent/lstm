# -*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
import sys
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

# 窗口长度
LEN_SEQ = 4

def load_one(data, x):
    global LEN_SEQ
    ticker = data.query("TICKER_SYMBOL==" + str(x)) # 个股
    arr = ticker.ix[:,[2,3,4,5]].values # 矩阵
    # 做时序差分
    train, label = [], []
    b_size = 4
    for i in range(LEN_SEQ,0,-1):
        train.append(arr.shift(i))
        label += [('var%d(t-%d)' % (j+1,i)) for j in range(b_size)]
    for i in range():
        train.append(arr.shift(-i))
        if i ==0:
            tabel += [('var%d(t)' %(j+1)) for j in range(b_size)]
        else:
            names += [('var%d(t+%d)'%(j+1,i)) for j in range(b_size)]
    
    return train, label, b_size

def build_model(data):
    model = Sequential()
    model.add(LSTM(50,input_shape=(data.shape[1],data.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae',optimizer='adam')
    return model 
    

def predict_point_by_point(model, data):
    train_X,train_y = data[:,:-1],data[:,-1]
    test_X,test_y = data[:,:-1],test[:,-1]
    train_X = train_X.reshape((train_X.shape[0],LEN_SEQ,train_X.shape[1]))
    test_X = test_X.reshape(test_X,shape[0],LEN_SEQ,test_X.shape[1])
    LSTM = model.fit(train_X,train_y,epochs=100,batch_size=100)
    return model.predict(test_X)

def main():
    dic = {}
    data = pd.read_csv("./data.csv", header=0)
    # 做minmax
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(data)
    
    for i in range(5): # 按股循环
        train, label, b_size = load_data(i)
        # 做模型
        model = build_model(data_scaled)
        # 做预测
        #dic[str(i)] = "预测数值"
        dic[str(i)] = predict_point_by_point(model,data_scaled)
if __name__ == "__main__":
    sys.exit(main())
