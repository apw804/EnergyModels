import tensorflow as tf

# create a TensorFlow operation that uses a GPU
with tf.device('/GPU:0'):
    a = tf.constant([1.0, 2.0, 3.0])
    b = tf.constant([4.0, 5.0, 6.0])
    c = a * b

# run the operation and print the result
print(c)
