from keras.models import Sequential
from keras.layers.core import Dense, Activation
import pandas as pd
import io
import requests
import numpy as np
from sklearn import metrics
from keras.layers import Dropout
from matplotlib.pyplot import figure, show
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np
from sklearn import metrics
from scipy.stats import zscore
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1] for red,green,blue)
def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)


# Encode text values to a single dummy variable.  The new columns (which do not replace the old) will have a 1
# at every location where the original column (name) matches each of the target_values.  One column is added for
# each target value.
def encode_text_single_dummy(df, name, target_values):
    for tv in target_values:
        l = list(df[name].astype(str))
        l = [1 if str(x) == str(tv) else 0 for x in l]
        name2 = "{}-{}".format(name, tv)
        df[name2] = l


# Encode text values to indexes(i.e. [1],[2],[3] for red,green,blue).
def encode_text_index(df, name):
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_


# Encode a numeric column as zscores
def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean) / sd


# Convert all missing values in the specified column to the median
def missing_median(df, name):
    med = df[name].median()
    df[name] = df[name].fillna(med)


# Convert all missing values in the specified column to the default
def missing_default(df, name, default_value):
    df[name] = df[name].fillna(default_value)


# Convert a Pandas dataframe to the x,y inputs that TensorFlow needs
def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(target_type, '__iter__') else target_type
    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(df[target])
        return pd.DataFrame(df,columns=result).to_numpy(dtype=np.float32), dummies.to_numpy(dtype=np.float32)
    else:
        # Regression
        return pd.DataFrame(df,columns=result).to_numpy(dtype=np.float32), pd.DataFrame(df,columns=[target]).to_numpy(dtype=np.float32)


# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


# Regression chart.
def chart_regression(pred,y,sort=True):
    t = pd.DataFrame({'pred' : pred, 'y' : y.flatten()})
    if sort:
        t.sort_values(by=['y'],inplace=True)
    a = plt.plot(t['y'].tolist(),label='expected')
    b = plt.plot(t['pred'].tolist(),label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()

# Remove all rows where the specified column is +/- sd standard deviations
def remove_outliers(df, name, sd):
    drop_rows = df.index[(np.abs(df[name] - df[name].mean()) >= (sd * df[name].std()))]
    df.drop(drop_rows, axis=0, inplace=True)


# Encode a column to a range between normalized_low and normalized_high.
def encode_numeric_range(df, name, normalized_low=-1, normalized_high=1,
                         data_low=None, data_high=None):
    if data_low is None:
        data_low = min(df[name])
        data_high = max(df[name])

    df[name] = ((df[name] - data_low) / (data_high - data_low)) \
               * (normalized_high - normalized_low) + normalized_low

def classify(a, b, c, d, e, f):
	df = pd.read_csv('coin_Cardano.csv')
	df.drop('SNo',1,inplace=True)
	df.drop('Name',1,inplace=True)
	df.drop('Symbol',1,inplace=True)
	df.Date = df.Date.str.replace('-','')
	df['Date'] = df['Date'].str.split(' ').str[0]
	df['Date'] = df['Date'].astype('int64')
	df_2 = {'Date': int(a), 'High': float(b), 'Low': float(c), 'Open': float(d), 'Close': int('2'), 'Volume': float(e), 'Marketcap': float(f)}
	df = df.append(df_2, ignore_index = True)
	encode_numeric_zscore(df, 'Volume')
	encode_numeric_zscore(df, 'Marketcap')
	encode_numeric_zscore(df, 'Date')
	x,y = to_xy(df,'Close')
	model = Sequential()
	model.add(Dense(100, input_dim=x.shape[1], activation='relu')) # Hidden 1
	model.add(Dense(10, activation='relu')) # Hidden 2
	model.add(Dense(1)) # Output
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(x,y,verbose=2,epochs=100)
	#x = np.append(x, np.array([[a, b, c, d, e, f]]), axis=0)
	#x = x.astype(np.float64)
	pred = model.predict(x)
	r = pred[-1]
	s = np.array2string(r)
	return s
       

