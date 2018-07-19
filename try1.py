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
df1 = pd.read_excel(filename1)
#from colors import red
