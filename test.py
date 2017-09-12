import tensorflow as tf
tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
FLAGS = tf.app.flags.FLAGS

print FLAGS.learning_rate