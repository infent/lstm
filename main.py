
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
np.random.seed(0)

def load_one(data, x):
    global LEN_SEQ
    #ticker = data.query("TICKER_SYMBOL=="+str(x)) # 个股
    #ticker = data.query("TICKER_SYMBOL==1 or TICKER_SYMBOL==5")
    ticker =data.loc[data['TICKER_SYMBOL']==x]
    #ticker = data.query("TICKER_SYMBOL==@x")
    print('query the data')
    #print(ticker)
    arr = ticker.ix[:,[1,2,3,4]] # 矩阵
   # print(arr)
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
    #Train.columns = label
    return Train

def build_model():
    model = Sequential()
    model.add(LSTM(20,input_shape=(1,15)))
    model.add(Dense(1))
    model.compile(loss='mae',optimizer='adam')
    return model 
    

def predict_point_by_point(model, data):
    values = data.values
    train_X,train_y = values[:,:-1],values[:,-1]
    #test_X,test_y = values[:,:-1],data[:,-1]
    train_X = train_X.reshape((train_X.shape[0],1,train_X.shape[1]))
    #test_X = test_X.reshape(test_X,shape[0],LEN_SEQ,test_X.shape[1])
    LSTM = model.fit(train_X,train_y,epochs=20,batch_size=3)
    return model

#在金融的这份数据里，没有2018年q2的数据，也就是t+1的cogs，operateprofit，nincome都没有，无法去预测目标，也就是revenue的值。所以我们需要先对每个单列做出预测，这里我们仍然用lstm对2018q2的这三列预测，再加上向前的三次记录，共十五列来做预测

def create_trainX_trainy(data, look_back=1):
    trainX, trainy = [], []
    for i in range(len(data)-look_back-1):
        a = data[i:(i+look_back)]
        trainX.append(a)
        trainy.append(data[i + look_back])
    return np.array(trainX), np.array(trainy)


def predict_useone_column(column):
    trainX,trainy = create_trainX_trainy(column)
    trainX = np.reshape(trainX,(trainX.shape[0],1,trainX.shape[1]))

    model = Sequential()
    model.add(LSTM(20,input_shape=(1,1)))
    model.add(Dense(1))
    model.compile(loss='mae',optimizer='adam')
        

    
    model.fit(trainX,trainy,epochs=20,batch_size=1)
    return model

def main():
    dic = {}
    data = pd.read_csv("./datanew.csv", header=0)
    data.drop('END_DATE',1,inplace=True)
    #print(data.head(),data.columns)
    
    data.fillna(0.00001,inplace=True)
    

    #print(temp.isnull().count())
    #temp.fillna(0.000001,inplace=True)
    # 做minmax
    #scaler = MinMaxScaler(feature_range=(0,1))
    #data_scaled = pd.DataFrame(scaler.fit_transform(temp),columns=['TICKER_SYMBOL','REVENUE','COGS','OPERATE_PROFIT','N_INCOME'])
    
    tickers = data['TICKER_SYMBOL'].unique()
    #train, label, b_size = [], [], []




    #按股训练，先出q2的前三列
    for i in tickers: # 按股循环
        print(i)
        train = load_one(data,i)
        if train.index.values ==[]:
            continue
        #print(train)
        
        
        
        #for j in [train.ix[:,12],train.ix[:,13],train.ix[:,14]]:
         #   predict_useone_column(model,column)
          #  train.ix[]
        #print(train.values.shape)
       # print(train.head())
        # 做模型
       # print(i)
        model = build_model()
        # 做预测
        #dic[str(i)] = "预测数值"
        #LSTM = predict_point_by_point(model,train)
        #print(data.loc[data['TICKER_SYMBOL']==i]['COGS'].values)
        cogs_model=predict_useone_column((data.loc[data['TICKER_SYMBOL']==i])['COGS'].values)
        operate_model=predict_useone_column((data.loc[data['TICKER_SYMBOL']==i])['OPERATE_PROFIT'].values)
        nincome_model=predict_useone_column((data.loc[data['TICKER_SYMBOL']==i])['N_INCOME'].values)
        
        
        
        pre2018q2 = np.array(list(train.iloc[-1,3:15].values))
        cogs = cogs_model.predict(np.reshape(np.array(pre2018q2[9]),(1,1,1)))
        operate = operate_model.predict(np.reshape(np.array(pre2018q2[10]),(1,1,1)))
        nincome = nincome_model.predict(np.reshape(np.array(pre2018q2[11]),(1,1,1)))
        pre2018q2 = np.append(pre2018q2,[cogs,operate,nincome])
        #print(pre2018q2)
        pre2018q2 = np.reshape(pre2018q2,(1,1,15))
    
        
        model_after_train = predict_point_by_point(model,train)
        pre_revenue = model_after_train.predict(pre2018q2)[0][0]
        #print(pre_revenue,len(pre_revenue))
        dic[str(i)] = pre_revenue
        #train = pd.DataFrame()
        
    result = pd.DataFrame(columns=['ticker_symbol','predict_revenue'])
    result['ticker_symbol'] = dic.keys()
    result['predict_revenue'] = dic.values()
    result.to_csv('./predict.csv')
    
if __name__ == "__main__":
    sys.exit(main())
