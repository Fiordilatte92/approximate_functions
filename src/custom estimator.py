import src.produce_data
import pandas as pd
import tensorflow as tf
import plotly as py
import plotly.graph_objs as go
import numpy as np


steps= 400
LEARNING_RATE=0.01
NUM_EPOCHS=2000
model_params = {"learning_rate": LEARNING_RATE}


#produce train and test data
src.produce_data.write_train_data(steps)
src.produce_data.write_test_data(steps)

dft=pd.read_csv("../input/train_data.csv" ,header=0, names=["x1","x2","tar"]  , index_col=None)
dft_x=dft.filter(items=['x1', 'x2'])
dfts=pd.DataFrame.from_csv("../input/test_data.csv")


#input train data

training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename="../input/train_data.csv", target_dtype=np.float32, features_dtype=np.float32)

train_input_fn2 = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(training_set.data)},
    y=np.array(training_set.target),
    shuffle=True,
    num_epochs=NUM_EPOCHS)
test_input_fn2 = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(training_set.data)},
    y=np.array(training_set.target),
    shuffle=False,
    )

# train_input_fn = tf.estimator.inputs.pandas_input_fn(
#     x=dft_x,
#     y=dft.y,
#     shuffle=True,
#     num_epochs=NUM_EPOCHS)
# test_input_fn = tf.estimator.inputs.pandas_input_fn(
#     x=dfts,
#     y=dfts.y,
#     shuffle=False,
#     num_epochs=NUM_EPOCHS)


def model_fn(features, labels, mode, params):
    """Model function for Estimator."""

    # Connect the first hidden layer to input layer
    # (features["x"]) with relu activation

    #first_hidden_layer = tf.layers.dense(input_layer, 30, activation=tf.nn.relu)
    first_hidden_layer = tf.layers.dense(tf.reshape(features["x"],shape=[-1,2]), 40, activation=tf.nn.relu)

    # Connect the second hidden layer to first hidden layer with relu
    second_hidden_layer = tf.layers.dense(first_hidden_layer, 30, activation=tf.nn.relu)

    # Connect the output layer to second hidden layer (no activation fn)
    output_layer = tf.layers.dense(second_hidden_layer, 1)

    # Reshape output layer to 1-dim Tensor to return predictions
    predictions = tf.reshape(output_layer, [-1])

    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions)

    # Calculate loss using mean squared error
    loss = tf.losses.mean_squared_error(labels, predictions)

    # Calculate root mean squared error as additional eval metric
    eval_metric_ops = {
        "rmse": tf.metrics.root_mean_squared_error(
            tf.cast(labels, tf.float32), predictions)
    }

    #optimizer = tf.train.GradientDescentOptimizer(
       # learning_rate=params["learning_rate"])
    optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])



    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)

nn=tf.estimator.Estimator(model_fn=model_fn, params=model_params )
nn.train(input_fn=train_input_fn2, steps=None)
ev= nn.evaluate(input_fn=test_input_fn2)

pre=nn.predict(input_fn=test_input_fn2)
pred= [ p for p in pre]
# def predict():
#     yp = nn.predict(input_fn=train_input_fn2)
#     yp_ = list(yp)
#     y_pred=[]
#     for i in range(steps):
#         y_pred.append(float(yp_[i]['approx']))
#     return y_pred

real_data = go.Scatter3d(x= dft.x1,
                         y= dft.x2,
                         z=dft.tar,
                         mode='markers')
pred_data = go.Scatter3d(x= dft.x1,
                       y=dft.x2,
                       z= pred,
                       mode='markers')
data=[real_data, pred_data]
py.offline.plot(data, filename='lel.html')


l = abs(dft.tar.values - pred)
l= np.sqrt(np.sum([x*x for x in l]))
print("manual loss %s" % l)
print("Loss: %s" % ev["loss"])
print("Root Mean Squared Error: %s" % ev["rmse"])