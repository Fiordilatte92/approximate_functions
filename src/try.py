import numpy as np
import tensorflow as tf

training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename="../input/train_data.csv", target_dtype=np.float32, features_dtype=np.float32)