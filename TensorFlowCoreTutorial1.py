# Getting Started With TensorFlow 1 - https://www.tensorflow.org/get_started/get_started
import tensorflow as tf

# Add two constant floating point tensors
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
# Output the tensors - note they are not evaluated yet
print(node1, node2)

# Output the values of the tensors
sess = tf.Session()
print(sess.run([node1, node2]))

# Combine the two tensors by using the 'add' operation, and output the resulting node and its value
node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))

# Parameterize the graph to accept external inputs, known as placeholders
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))
print(sess.run(adder_node, {a: [1, 3, 5], b: [2, 4, 6]}))

# Make the computational graph more complex by adding another operation
add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))
print(sess.run(add_and_triple, {a: [1, 3], b: [2, 4]}))
print(sess.run(add_and_triple, {a: [1, 3, 5], b: [2, 4, 6]}))




