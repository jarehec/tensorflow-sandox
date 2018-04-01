import tensorflow as tf

with tf.Graph().as_default():
    matrix = tf.constant([[3,4,5,6], [6,4,2,8], [0,4,2,8]], dtype=tf.int32)
    reshaped_matrix = tf.reshape(matrix, [6,2])
    with tf.Session() as sess:
        print(matrix.eval())
        print(reshaped_matrix.eval())
