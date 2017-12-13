import src.produce_data
import pandas as pd
import tensorflow as tf
import plotly as py
import plotly.graph_objs as go
import itertools
from functools import reduce
import numpy as np
steps= 100


#produce train and test data
src.produce_data.write_train_data(steps)
src.produce_data.write_test_data(steps)

dft=pd.DataFrame.from_csv("../input/train_data.csv")
dfts=pd.DataFrame.from_csv("../input/test_data.csv")


#input train data

train_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=dft,
    y=dft.y,
    shuffle=True)

feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]
# optimizer=tf.estimator.LinearRegressor(feature_columns)

optimizer = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=tf.train.FtrlOptimizer(
        learning_rate=10,
        l1_regularization_strength=0.001
    ))
optimizer.train(train_input_fn)

#test data 
test_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=pd.DataFrame(dfts.x),
    y=dfts.y,
    shuffle=False)


def predict():
    yp = optimizer.predict(input_fn=test_input_fn)
    yp_ = list(yp)
    y_pred=[]
    for i in range(steps):
        y_pred.append(float(yp_[i]['predictions']))
    return y_pred


trace0 = go.Scatter( x= dft.x,
                     y= dft.y,
                     mode='markers')
trace1 = go.Scatter( x= dfts.x,
                     y= predict(),
                     mode='markers')
data=[trace0, trace1]
py.offline.plot(data, filename='lel')


# Specify that all features have real-value data
#feature_columns = [tf.feature_column.numeric_column("x", shape=[2])]
