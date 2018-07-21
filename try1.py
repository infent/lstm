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
col_temp = ['TICKER_SYMBOL','END_DATE','SELL_EXP','ADMIN_EXP','N_INCOME']
tempdata = pd.DataFrame(data,columns = col_temp).head(10000)
'''
result analyse:
 - 0s - loss: 0.0427 - val_loss: 0.0061
Epoch 2/100
 - 0s - loss: 0.0084 - val_loss: 0.0042
Epoch 3/100
 - 0s - loss: 0.0077 - val_loss: 0.0043
上来loss就低的一比,这是为什么?
预测:
[0.12347972]
 [0.12208527]
 [0.12162907]
实际:
[0.12360479 0.12157936 0.12182428]
原因与我们归一化了有很大关系,我们预测的数量级在e8-11级别即使预测值看似很接近但是差距在万或十万级别,但是这也太准了吧,怎么回事
后来又想了想,利润这个东西本来季度间除了个别行业差距也不大,用t-1预测t是否有失偏颇?
'''
'''
    TICKER_SYMBOL    END_DATE      SELL_EXP     ADMIN_EXP      N_INCOME
0               2  2009-03-31  2.385783e+08  2.828142e+08  8.885426e+08
1               2  2011-09-30  1.689365e+09  1.271355e+09  4.106349e+09
2               2  2013-06-30  1.431103e+09  1.192209e+09  5.335891e+09
3               2  2014-12-31  4.521889e+09  3.902618e+09  1.928752e+10
4               2  2014-12-31  4.521889e+09  3.902618e+09  1.928752e+10
5               2  2015-03-31  6.872442e+08  7.128519e+08  9.080262e+08
6               2  2016-09-30  3.327747e+09  3.171476e+09  1.129025e+10
7               4  2008-12-31  5.592508e+06  1.623657e+07 -1.148419e+07
8               4  2008-12-31  5.592508e+06  1.623657e+07 -1.148419e+07
9               4  2009-06-30  2.685072e+06  7.489249e+06 -8.338033e+05
10              4  2009-12-31  5.788340e+06  1.579437e+07  4.532534e+06
11              2  2008-09-30  1.133777e+09  1.003706e+09  2.634320e+09
12              2  2008-12-31  1.860350e+09  1.530799e+09  4.639869e+09
13              2  2009-12-31  1.513717e+09  1.441987e+09  6.430008e+09
14              2  2010-03-31  2.932904e+08  3.092530e+08  1.175955e+09
15              2  2010-03-31  2.932904e+08  3.092530e+08  1.175955e+09
16              2  2012-03-31  5.822364e+08  5.064567e+08  1.530547e+09
17              2  2012-03-31  5.822364e+08  5.064567e+08  1.530547e+09
18              2  2012-09-30  2.104936e+09  1.596507e+09  6.146172e+09
19              2  2012-12-31  3.056378e+09  2.780308e+09  1.566259e+10
20              2  2013-03-31  6.233992e+08  5.727349e+08  1.789369e+09
21              2  2013-12-31  3.864714e+09  3.002838e+09  1.829755e+10
22              2  2009-03-31  2.385783e+08  2.828142e+08  8.885426e+08
23              2  2011-03-31  3.935214e+08  5.626788e+08  1.189965e+09
24              2  2011-06-30  9.567482e+08  8.702434e+08  3.252518e+09
25              2  2012-12-31  3.056378e+09  2.780308e+09  1.566259e+10
26              2  2015-03-31  6.872442e+08  7.128519e+08  9.080262e+08
27              2  2015-12-31  4.138274e+09  4.745250e+09  2.594944e+10
28              2  2017-06-30  2.102697e+09  2.604623e+09  1.005299e+10

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
    print(agg.columns)
    #丢弃被错位行
    #Index(['var1(t-1)', 'var2(t-1)', 'var3(t-1)', 'var1(t)', 'var2(t)', 'var3(t)'], dtype='object')
    if dropnan:
        agg.dropna(inplace=True)
    print(agg.head())
    return agg
    
#数据归一化
#tempdata.iloc[:,[2,3,4]]
scaler = MinMaxScaler(feature_range=(0,1))
#scaled_data =scaler.fit_transform(tempdata.iloc[:,[2,3,4]])
#将时序数据转化为监督问题数据
tempdata1 = tempdata.iloc[:,[2,3,4]]
tempdata1.fillna(0.0,inplace=True)
tempdata1 = scaler.fit_transform(tempdata1)
#print(tempdata1.values)
reframed = series_to_supervised(tempdata1,1,1)
print(reframed.values)
#删除无用的label
#reframed.drop(reframed.columns[[6,7,8,9]],axis=1,inplace=True)





#划分数据集
train_days = 5500
valid_days = 1500
values = reframed.values
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
#(5500, 1, 5) (5500,) (1500, 1, 5) (1500,) (2155, 1, 5) (2155,)

#建立模型训练
model1 = Sequential()
#输入层为50个cell组成的lstm,接受时间步长为1,特征数为5的输入（这里有疑问，为什么本来要作为预测项的t-1的Nincome也要作为特征？）
model1.add(LSTM(50,input_shape=(train_X.shape[1],train_X.shape[2])))
#接了一个全连接层,来输出预测值
model1.add(Dense(1))
model1.compile(loss='mae',optimizer='adam')
#model1.add(LSTM(50,activation='relu',input_shape=(train_X.shape[1],train_X.shape[2]),return_sequences=True))
#如果returnsequences=true 返回【samples,output_dims】else return [samples,step,output_dims]
#model1.add(LSTM(50,activation='relu',input_shape=(train_X.shape[1],train_X.shape[2]))
#model1.add(Dense(1,activation='linear'))
#model1.compile(loss='mean_squared_error',optimizer='adam')

#拟合
LSTM = model1.fit(train_X,train_y,epochs=100,batch_size=100,validation_data = (valid_X,valid_y),verbose=2, shuffle=False)
print(model1.evaluate(test_X,test_y))
print('-------------------------------------------------------------')
print(str(model1.predict(test_X))+'=================='+str(test_y))
plt.plot(LSTM.history['loss'],label='train')
plt.plot(LSTM.history['val_loss'],label='valid')
#显示图例
plt.legend()
plt.show()


'''
plt.figure(figsize=(3000,3))
train_predict = model1.predict(train_X)
valid_predict = model1.predict(valid_X)
test_predict = model1.predict(test_X)
plt.plot(values[:,-1],c='b')
plt.plot([x for x in train_predict],c='g')
plt.plot([None for _ in train_predict] + [x for x in valid_predict],c='y')
plt.plot([None for _ in train_predict] + [None for _ in valid.predict] + [x for x in test_predict],c='r')
plt.show()
'''
