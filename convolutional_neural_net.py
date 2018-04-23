# Convolution -> Pooling -> Convolution -> Pooling
# -> Fully Connected Layer -> Output
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)

batch_size = 128
n_classes = 10
n_channels = 1
num_epochs = 10

def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

def max_pool_2d(x):
    return tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=2)

def convolutional_model_fn(features, labels, mode, params):
    x = features['x']
    cnn = tf.reshape(x, shape=[-1, img_size, img_size, n_channels])
    # conv1 -> pool1 -> conv2 -> pool2
    cnn = tf.layers.conv2d(inputs=cnn, name='conv1_layer',
                           filters=16, kernel_size=4,
                           padding='SAME', activation=tf.nn.relu)
    cnn = max_pool_2d(cnn)
    cnn = tf.layers.conv2d(inputs=cnn, name='conv2_layer',
                           filters=36, kernel_size=4,
                           padding='SAME', activation=tf.nn.relu)
    cnn = max_pool_2d(cnn)
    # flatten cnn
    cnn = tf.layers.flatten(cnn)
    # fully-connected layer
    cnn = tf.layers.dense(inputs=cnn, name='fc1_layer',
                          units=128, activation=tf.nn.relu)
    # second fully-connected layer
    cnn = tf.layers.dense(inputs=cnn, name='fc2_layer', units=10)
    # logits output of cnn
    logits = cnn
    # Softmax output of the neural network.
    prediction = tf.nn.softmax(logits=logits)
    # Classification output of the neural network.
    pred_class = tf.argmax(prediction, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode, predictions=pred_class)
    else:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels)
        loss = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        # Eval metrics
        eval_metrics = {'accuracy': tf.metrics.accuracy(labels=labels,
                                                        predictions=pred_class)}
        spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          train_op=train_op,
                                          eval_metric_ops=eval_metrics)
    return spec


def main(_):
    mnist.train.cls = np.argmax(mnist.train.labels, axis=1)
    mnist.test.cls = np.argmax(mnist.test.labels, axis=1)
    mnist.validation.cls = np.argmax(mnist.validation.labels, axis=1)
    some_images = mnist.validation.images[0:9]
    params = {'learning_rate': 1e-3}
    
    # Get the first images from the test-set.
    images = mnist.test.images[0:9]
    # Get the true classes for those images.
    cls_true = mnist.test.cls[0:9]
    # Plot the images and labels using our helper-function above.
    plot_images(images=images, cls_true=cls_true)
    
    mnist_classifier = tf.estimator.Estimator(model_fn=convolutional_model_fn,
                                              params=params,
                                              model_dir='/tmp/mnist_cnn')
    # train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': np.array(mnist.train.images)},
                                                        y=np.array(mnist.train.cls),
                                                        batch_size=batch_size,
                                                        num_epochs=None,
                                                        shuffle=True)

    # evaluate the model and print results
    test_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': np.array(mnist.test.images)},
                                                       y= np.array(mnist.test.cls),
                                                       num_epochs=1,
                                                       shuffle=False)
    # predict class
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": some_images},
                                                          num_epochs=1,
                                                          shuffle=False)
    
    mnist_classifier.train(input_fn=train_input_fn, steps=500)
    test_results = mnist_classifier.evaluate(input_fn=test_input_fn)
    print(test_results)
    predictions = mnist_classifier.predict(input_fn=predict_input_fn)
    cls_pred = np.array(list(predictions))
    print(cls_pred)

    plot_images(images=some_images,
            cls_true=mnist.validation.cls[:9],
            cls_pred=cls_pred)
    
if __name__ == '__main__':
    tf.app.run()