import tensorflow as tf
import pandas as pd
import src.produce_data
import numpy as np

steps=100
#produce train and test data
src.produce_data.write_train_data(steps)
src.produce_data.write_test_data(steps)



dft=pd.DataFrame.from_csv("../input/train_data.csv")
dfts=pd.DataFrame.from_csv("../input/test_data.csv")

x= tf.placeholder(tf.float32, [None, 1])
W=tf.Variable(tf.zeros([1,20]))
b= tf.Variable(tf.zeros([20]))
y=tf.nn.softmax(tf.matmul(x,W)+b)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
    batch_x = dftA
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})