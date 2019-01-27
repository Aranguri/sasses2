import tensorflow as tf
import opennmt as onmt

tf.enable_eager_execution()

# Build a random batch of input sequences.
inputs = tf.random_uniform((3, 6, 256))
sequence_length = [1000, 1000, 1000]

# Encode with a self-attentional encoder.
encoder = onmt.encoders.SelfAttentionEncoder(num_layers=4)
# encoder = onmt.encoders.MeanEncoder()
outputs, state, outputs_length = encoder.encode(
    inputs,
    sequence_length=sequence_length,
    mode=tf.estimator.ModeKeys.TRAIN)

print(state)
