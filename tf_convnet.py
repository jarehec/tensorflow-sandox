import numpy as np
import tensorflow as tf


def tf_convolutional_neural_net(features, labels, mode):
    """
    Convolutional Layer #1: Applies 32 5x5 filters (extracting 5x5-pixel subregions), with ReLU activation function
    Pooling Layer #1: Performs max pooling with a 2x2 filter and stride of 2 (which specifies that pooled regions do not overlap)
    Convolutional Layer #2: Applies 64 5x5 filters, with ReLU activation function
    Pooling Layer #2: Again, performs max pooling with a 2x2 filter and stride of 2
    Dense Layer #1: 1,024 neurons, with dropout regularization rate of 0.4 (probability of 0.4 that any given element will be dropped during training)
    Dense Layer #2 (Logits Layer)
    """
    input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])
    convolutional_layer_1 = tf.layers.conv2d(inputs=input_layer,
                                             filters=32,
                                             kernel_size=[5,5],
                                             padding='SAME',
                                             activation=tf.nn.relu)

    pooling_layer_1 = tf.layers.max_pooling2d(inputs=convolutional_layer_1,
                                              pool_size=[2,2],
                                              strides=2)

    convolutional_layer_2 = tf.layers.conv2d(inputs=pooling_layer_1,
                                             filters=64,
                                             kernel_size=[5,5],
                                             padding='SAME',
                                             activation=tf.nn.relu)

    pooling_layer_2 = tf.layers.max_pooling2d(inputs=convolutional_layer_2,
                                              pool_size=[2,2],
                                              strides=2)
    # flattened feature map
    pooling_layer_flat = tf.reshape(pooling_layer_2, [-1, 7 * 7 * 64])

    dense_layer = tf.layers.dense(inputs=pooling_layer_flat,
                                  units=1024,
                                  activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=dense_layer,
                                rate=0.4,
                                training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer, 10 classes
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
    # Generate predictions (for PREDICT and EVAL mode)
    'classes': tf.argmax(input=logits, axis=1),
    # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
    # `logging_hook`.
    'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=labels,
                                        predictions=predictions['classes'])}
    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops)

def main(unsed):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    mnist_conv_nn = tf.estimator.Estimator(model_fn=tf_convolutional_neural_net,
                                           model_dir='mnist_convnet_model')

    # Set up logging for predictions
    tensors_to_log = {'probabilities': 'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                              every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': train_data},
                                                        y=train_labels,
                                                        batch_size=100,
                                                        num_epochs=None,
                                                        shuffle=True)
    mnist_conv_nn.train(input_fn=train_input_fn,
                        steps=20000,
                        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': eval_data},
                                                       y=eval_labels,
                                                       num_epochs=1,
                                                       shuffle=False)
    eval_results = mnist_conv_nn.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == '__main__':
  tf.app.run()