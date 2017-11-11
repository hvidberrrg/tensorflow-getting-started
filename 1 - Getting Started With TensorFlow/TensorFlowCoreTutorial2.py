# Getting Started With TensorFlow 2 - https://www.tensorflow.org/get_started/get_started
import tensorflow as tf

# Model parameters - variables constructed with a type and initial value:
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# Model input - a placeholder is a promise to provide a value later.
x = tf.placeholder(tf.float32)
# Define the model
linear_model = W*x + b

# In order to evaluate the model on training data, we need a y placeholder to provide the desired result values
y = tf.placeholder(tf.float32)
# ... and we need a loss function to measure how far apart the current model is from the provided result
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squared deltas

# Optimizers slowly change each variable in order to minimize the loss function
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# Initialize all the variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Define the training data
input = [1, 2, 3, 4]
result = [0, -1, -2, -3]
# Evaluate 'linear_model' for several values of x simultaneously
print(sess.run(linear_model, {x: input}))  # =>  [ 0.   0.30000001  0.60000002  0.90000004]
# Print the 'loss' function for these values
print(sess.run(loss, {x: input, y: result})) # ((.3*1-.3)-0)^2 + ((.3*2-.3)+1)^2 + ((.3*3-.3)+2)^2 + ((.3*4-.3)+3)^2  =  23.66
# Do the training loop
for i in range(1000):
    sess.run(train, {x: input, y: result})

# Output training accuracy - values should be: W ~ -1, b ~ 1
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: input, y: result})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))