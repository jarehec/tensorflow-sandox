import tensorflow as tf

with tf.Graph().as_default(), tf.Session() as sess:
    t = tf.Variable(tf.random_uniform(shape=[10,1], minval=1, maxval=7, dtype=tf.int32))
    v = tf.Variable(tf.random_uniform(shape=[10,1], minval=1, maxval=7, dtype=tf.int32))
    tv = tf.add(t, v)
    result = tf.concat([t, v, tv], 1)
    sess.run(tf.global_variables_initializer())
    print(t.eval())
    print(v.eval())
    print(result.eval())
