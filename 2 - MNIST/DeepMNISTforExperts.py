import tensorflow as tf
import tempfile

# Import the MNIST data set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) # A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension (used for labels, 0-9)

# Define a few helper functions
def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1) # initialize weights with a small amount of noise for symmetry breaking, and to prevent 0 gradients
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride - Our convolutions uses a stride of one and are zero padded so that the output is the same size as the input"""
    return tf.nn.conv2d(x,
                        W,
                        strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X - Our pooling is plain old max pooling over 2x2 blocks. """
    return tf.nn.max_pool(x,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')

# Create nodes for the input images (x) and target output (y') classes.
x = tf.placeholder(tf.float32, shape=[None, 784]) # placeholder -- a value that we'll input when we ask TensorFlow to run a computation.
y_ = tf.placeholder(tf.float32, shape=[None, 10])


# First convolutional layer - maps one grayscale image to 32 feature maps.
W_conv1 = weight_variable([5, 5, 1, 32]) # the convolution will compute 32 features for each 5x5 patch. Its weight tensor will have a shape of [5, 5, 1, 32]. The first two dimensions are the patch size, the next is the number of input channels, and the last is the number of output channels
b_conv1 = bias_variable([32]) # a bias vector with a component for each output channel
x_image = tf.reshape(x, [-1, 28, 28, 1]) # To apply the layer, we first reshape x to a 4d tensor, with the second and third dimensions corresponding to image width and height, and the final dimension corresponding to the number of color channels
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # convolve x_image with the weight tensor, add the bias, apply the ReLU function
h_pool1 = max_pool_2x2(h_conv1) # Max pool. The max_pool_2x2 method will reduce the image size to 14x14.

# Second convolutional layer -- maps 32 feature maps to 64.
W_conv2 = weight_variable([5, 5, 32, 64]) # The second layer will have 64 features for each 5x5 patch.
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Fully connected layer 1 -- after 2 rounds of downsampling, our 28x28 image
# is down to 7x7x64 feature maps -- maps this to 1024 features.
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout - To reduce overfitting, we will apply dropout before the readout layer
keep_prob = tf.placeholder(tf.float32) # We create a placeholder for the probability that a neuron's output is kept during dropout. This allows us to turn dropout on during training, and turn it off during testing
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) # TensorFlow's tf.nn.dropout op automatically handles scaling neuron outputs in addition to masking them, so dropout just works without any additional scaling

# Readout Layer - Map the 1024 features to 10 classes, one for each digit
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# Specify the loss function. Loss indicates how bad the model's prediction was on a single example, and we try to minimize it as we train
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

# Train the model, minimizing the loss function
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# Define the accuracy
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Save summary data to a temporary location
graph_location = tempfile.mkdtemp()
print('Saving graph to: %s' % graph_location)
train_writer = tf.summary.FileWriter(graph_location)
train_writer.add_graph(tf.get_default_graph())

# Do the actual training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0],
                y_: batch[1],
                keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


# Evaluate the model
print('test accuracy %g' % accuracy.eval(feed_dict={
    x: mnist.test.images,
    y_: mnist.test.labels,
    keep_prob: 1.0}))