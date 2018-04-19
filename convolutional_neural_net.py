# Convolution -> Pooling -> Convolution -> Pooling
# -> Fully Connected Layer -> Output
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

n_classes = 10
batch_size = 128

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, n_classes])

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def convolutional_neural_net(data):
    weights = {
        # 5x5 convolution, 1 input, 32 outputs
        'W_conv1': tf.Variable(tf.random_normal([5,5,1,32])),
        # 5x5 convolution, 32 inputs, 64 outputs
        'W_conv2': tf.Variable(tf.random_normal([5,5,32,64])),
        # fully-connected, 7*7*64 inputs, 1024 outputs
        'W_fc': tf.Variable(tf.random_normal([7*7*64, 1024])),
        # 1024 inputs, n_classes outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }
    biases = {
        'b_conv1': tf.Variable(tf.random_normal([32])),
        'b_conv2': tf.Variable(tf.random_normal([64])),
        'b_fc': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    data = tf.reshape(data, shape=[-1, 28, 28, 1])

    conv1 = tf.nn.relu(conv2d(data, weights['W_conv1']) + biases['b_conv1'])
    conv1 = max_pool_2d(conv1)
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = max_pool_2d(conv2)
    # Reshape conv2 output to fit fully connected layer
    fully_connected = tf.reshape(conv2, [-1,7*7*64])
    fully_connected = tf.nn.relu(tf.matmul(fully_connected, weights['W_fc']) + biases['b_fc'])
    output = tf.matmul(fully_connected, weights['out']) + biases['out']
    return output

def train_neural_network(x):
    prediction = convolutional_neural_net(x)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    n_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, loss], feed_dict={x:epoch_x, y:epoch_y})
                epoch_loss += c
            # TODO print batch completion
            print('Epoch', epoch + 1, 'completed out of', n_epochs, '\nloss:', epoch_loss)
        # This will tell us how many predictions we made that were perfect matches to their labels.
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)