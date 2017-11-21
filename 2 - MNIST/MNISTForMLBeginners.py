import tensorflow as tf

# Import the MNIST data set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) # A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension (used for labels, 0-9)

#####################
##
## DEFINE THE MODEL
##
#####################
# Define a placeholder for an image (a placeholder is a value that we'll input when we ask TensorFlow to run a computation)
x = tf.placeholder(tf.float32, [None, 784]) # None means that a dimension can be of any length, i.e. we can have any number of 28X28 pixel images
# Define variables for weights and biases (a Variable is a modifiable tensor that lives in TensorFlow's graph of interacting operations)
W = tf.Variable(tf.zeros([784, 10])) # W has a shape of [784, 10] because we want to multiply the 784-dimensional image vectors by it to produce 10-dimensional vectors of evidence for the different classes
b = tf.Variable(tf.zeros([10])) # b has a shape of [10] so we can add it to the output

# Define the model (y is the predicted label for the image, i.e. a one-hot vector representing a value between 0 and 9)
y = tf.nn.softmax(tf.matmul(x, W) + b) # multiply x by W, add b, and finally apply softmax
# If you want to assign probabilities to an object being one of several different things, softmax is the thing to do,
# because softmax gives us a list of values between 0 and 1 that add up to 1

#####################
##
## TRAIN THE MODEL
##
#####################
# In order to train the model we first need to define how far off our model is from our desired outcome - this is called the loss.
# By training we try to minimize the loss.
# y , defined above, is our predicted probability distribution, and y′ is the true distribution.
# First add a new placeholder to input the correct answers y':
y_ = tf.placeholder(tf.float32, [None, 10])
# Then we can implement the cross-entropy function, −∑y′log(y), measuring how inefficient our predictions are for describing the truth.:
# The raw formulation, tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])), of cross-entropy can be numerically unstable.
# Instead we use:
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# We can now launch the model in an InteractiveSession:
sess = tf.InteractiveSession()
# Initialize the variables we created:
tf.global_variables_initializer().run()
# Run the training step 1000 times!
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100) # get a "batch" of one hundred random data points from our training set
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) # run train_step feeding in the batches data to replace the placeholders.

#####################
##
## EVALUATE THE MODEL
##
#####################
# In order to evaluate the model we use tf.argmax. tf.argmax(y,1) is the label our model thinks is most likely for each input, while tf.argmax(y_,1) is the correct label
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# Cast the vector of booleans to floating point numbers and then take the mean to get a fraction that represents the accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Finally, we ask for the accuracy on our test data.
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))










