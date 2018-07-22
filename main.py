# -*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
import sys
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

# 窗口长度
LEN_SEQ = 2

def load_one(data, x):
    global LEN_SEQ
    #ticker = data.query("TICKER_SYMBOL=="+str(x)) # 个股
    ticker = data.query("TICKER_SYMBOL==@x")
    arr = ticker.ix[:,[1,2,3,4]] # 矩阵
    # 做时序差分
    train, label = [], []
    b_size = 4
    for i in range(LEN_SEQ,0,-1):
        train.append(arr.shift(i))
        label += [('var%d(t-%d)' % (j+1,i)) for j in range(b_size)]
    for i in range(LEN_SEQ):
        train.append(arr.shift(-i))
        if i ==0:
            label += [('var%d(t)' %(j+1)) for j in range(b_size)]
        else:
            label += [('var%d(t+%d)'%(j+1,i)) for j in range(b_size)]
    Train = pd.concat(train,axis=1)
    Train.dropna(inplace=True)
    Train.columns = label
    return Train, label, b_size

def build_model(data):
    model = Sequential()
    model.add(LSTM(50,input_shape=(1,15)))
    model.add(Dense(1))
    model.compile(loss='mae',optimizer='adam')
    return model 
    

def predict_point_by_point(model, data):
    values = data.values
    train_X,train_y = values[:,:-1],values[:,-1]
    #test_X,test_y = values[:,:-1],data[:,-1]
    train_X = train_X.reshape((train_X.shape[0],1,train_X.shape[1]))
    #test_X = test_X.reshape(test_X,shape[0],LEN_SEQ,test_X.shape[1])
    LSTM = model.fit(train_X,train_y,epochs=100,batch_size=100)
    return model.predict(train_X)

def main():
    dic = {}
    data = pd.read_csv("./data.csv", header=0)
    data.drop('END_DATE',1,inplace=True)
    print(data.head(),data.columns)
    
    data.fillna(0.00001,inplace=True)
    
    temp = data
    #print(temp.isnull().count())
    #temp.fillna(0.000001,inplace=True)
    # 做minmax
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = pd.DataFrame(scaler.fit_transform(temp),columns=['TICKER_SYMBOL','REVENUE','COGS','OPERATE_PROFIT','N_INCOME'])
    
    for i in data['TICKER_SYMBOL'].unique(): # 按股循环
        train, label, b_size = load_one(data_scaled,i)
       # print(train.values.shape)
       # print(train.head())
        # 做模型
        model = build_model(train)
        # 做预测
        #dic[str(i)] = "预测数值"
        dic[str(i)] = predict_point_by_point(model,train)
if __name__ == "__main__":
    sys.exit(main())
