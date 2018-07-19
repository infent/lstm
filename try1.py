import pandas as pd
import numpy as py
from keras.models import Sequential
from keras.layers import Dense,LSTM, Dropout
import matplotlib.pyplot as plt
#% matplotlib inline
import glob,os
import seaborn as sns
import sys
from sklearn.preprocessing import MinMaxScaler
#columns = {}
print('reading data')
filename1 = './IncomeStatement.xls'
data = pd.read_excel(filename1)
#from colors import red
'''
Index(['PARTY_ID', 'TICKER_SYMBOL', 'EXCHANGE_CD', 'PUBLISH_DATE',
       'END_DATE_REP', 'END_DATE', 'REPORT_TYPE', 'FISCAL_PERIOD',
       'MERGED_FLAG', 'T_REVENUE', 'REVENUE', 'INT_INCOME', 'PREM_EARNED',
       'COMMIS_INCOME', 'SPEC_TOR', 'ATOR', 'T_COGS', 'COGS', 'INT_EXP',
       'COMMIS_EXP', 'PREM_REFUND', 'N_COMPENS_PAYOUT', 'RESER_INSUR_CONTR',
       'POLICY_DIV_PAYT', 'REINSUR_EXP', 'BIZ_TAX_SURCHG', 'SELL_EXP',
       'ADMIN_EXP', 'FINAN_EXP', 'ASSETS_IMPAIR_LOSS', 'SPEC_TOC', 'ATOC',
       'F_VALUE_CHG_GAIN', 'INVEST_INCOME', 'A_J_INVEST_INCOME', 'FOREX_GAIN',
       'OTH_EFFECT_OP', 'ASSETS_DISP_GAIN', 'AE_EFFECT_OP', 'OTH_GAIN',
       'OPERATE_PROFIT', 'NOPERATE_INCOME', 'NOPERATE_EXP', 'NCA_DISPLOSS',
       'OTH_EFFECT_TP', 'AE_EFFECT_TP', 'T_PROFIT', 'INCOME_TAX',
       'OTH_EFFECT_NP', 'AE_EFFECT_NP', 'N_INCOME', 'GOING_CONCERN_NI',
       'QUIT_CONCERN_NI', 'N_INCOME_ATTR_P', 'N_INCOME_BMA', 'MINORITY_GAIN',
       'OTH_EFFECT_NPP', 'AE_EFFECT_NPP', 'BASIC_EPS', 'DILUTED_EPS',
       'OTH_COMPR_INCOME', 'OTH_EFFECT_CI', 'AE_EFFECT_CI', 'T_COMPR_INCOME',
       'COMPR_INC_ATTR_P', 'COMPR_INC_ATTR_M_S', 'OTH_EFFECT_PCI',
       'AE_EFFECT_PCI'],
      dtype='object')

'''
def series_to_supervised(data, n_in=1,n_out=1,dropnan=True):
    #统计输入数据列数（特征数）
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(),list()

    #输入的序列为时间t前n-1份数据
    #倒序向cols中插入向下移动了n_in-1的错位dataframe,作为模型的输入序列,被错位的填充nan
    for i in range(n_in,0,-1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1,i)) for j in range(n_vars)]

    #预测序列就是向cols中填入由下向上移动的i的dataframe
    for i in range(0,n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)'%(j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)'%(j+1,i)) for j in range(n_vars)]


    #把输入序列和预测序列合并到一起
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    #丢弃被错位行
    if dropnan:
        agg.dropna(inplace=True)
    return agg
    
#数据归一化
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data =scaler.fit_transfor(example[['c1','c2','c3']])
#将时序数据转化为监督问题数据
reframed = series_to_supervised(scaled_data,1,1)
#删除无用的label
reframed.drop(reframed.columns[[6,7,8,9]],axis=1,inplace=True)





#划分数据集
train_days = 400
valid_days = 150
values = redf.values
train = values[:train_days, :]
valid = values[train_days:train_days+valid_days,:]
test = values[train_days+valid_days:,:]
train_X, train_y = train[:,:-1], train[:,-1]
valid_X,valid_y = valid[:,:-1], valid[:,-1]
test_X,test_y = test[:,:-1],test[:,-1]

#将数据集改造成符合lstm需要的数据格式【样本，时间步，特征】
train_X = train_X.reshape((train_X.shape[0],1,train_X.shape[1]))
valid_X = valid_X.reshape((valid_X.shape[0],1,valid_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0],1,test_X.shape[1]))
print(train_X.shape, train_y.shape,valid_X.shape,valid_y.shape,test_X.shape,test_y.shape)


#建立模型训练
model1 = Sequential()
model1.add(LSTM(50,activation='relu',input_shape=(train_X.shape[1],train_X.shape[2]),return_sequences=True))
model1.add(Dense(1,activation='linear'))
model1.compile(loss='mean_squared_error',optimizer='adam')

#拟合
LSTM = model.fit(train_X,train_y,epochs=100,batch_size=32,validation_data = (valid_X,valid_y),verbose=2, shuffle=False)
plt.plot(LSTM.LSTM['loss'],label='train')
plt.plot(LSTM.LSTM['val_loss'],label='valid')
#显示图例
plt.legend()
plt.show()



#模型预测并可视化
plt.figure(figsize=(24,8))
train_predict = model.predict(train_X)
valid_predict = model.predict(valid_X)
test_predict = model.predict(test_X)
plt.plot(values[:,-1],c='b')
plt.plot([x for x in train_predict],c='g')
plt.plot([None for _ in train_predict] + [x for x in valid_predict],c='y')
plt.plot([None for _ in train_predict] + [None for _ in valid.predict] + [x for x in test_predict],c='r')
plt.show()
