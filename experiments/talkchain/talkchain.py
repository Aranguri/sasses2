import sys
sys.path.append('../../')
from tasks.sentences.sen_task import SenTask
from utils.embedder.embedder import Embedder
import itertools
import tensorflow as tf
from util import *

batch_size = 128
hidden_size = 512
learning_rate = 1e-4
debug_steps = 3
embeddings_size = 50 # It's fixed from glove
source = 'GPU'

if source == 'GPU': LSTM = tf.contrib.cudnn_rnn.CudnnLSTM
else: LSTM = tf.contrib.rnn.LSTMBlockCell
task = SenTask(batch_size)
vocab_size = task.get_lengths()

embeddings_init = tf.placeholder(tf.float32, (vocab_size, embeddings_size))
sentences_ids = tf.placeholder(tf.int32, (batch_size, None))
sentences_hot = tf.one_hot(sentences_ids, vocab_size)
embeddings = tf.get_variable('embeddings', initializer=embeddings_init)
start_vector = tf.get_variable('start_vector', (1, embeddings_size))
projection = tf.get_variable('projection', (hidden_size, vocab_size))
inputs = tf.nn.embedding_lookup(embeddings, sentences_ids)

start_stacked = tf.tile(start_vector, [batch_size, 1])
start_stacked = tf.expand_dims(start_stacked, 1)
shifted_inputs = tf.concat((start_stacked, inputs[:, 1:]), axis=1)

# maydo: use gpu-lstm
rnn_fw = LSTM(num_units=hidden_size, name='rnn_fw')
rnn_bw = LSTM(num_units=hidden_size, name='rnn_bw')
_, final_state = tf.nn.dynamic_rnn(rnn_fw, inputs, dtype=tf.float32)
outputs, _ = tf.nn.dynamic_rnn(rnn_fw, shifted_inputs, dtype=tf.float32)
# Other options for the following function: (1) keep generating a prob. dist. directly, but use something more complex, like a mlp. (2) produce a vector in embedding space and compute the similarity over all the memories (aka attention.) Then we have a probability distribution.
outputs = tf.einsum('ijk,kl->ijl', outputs, projection)

loss = tf.losses.softmax_cross_entropy(sentences_hot, outputs)
optimizer = tf.train.AdamOptimizer(learning_rate)
minimize = optimizer.minimize(loss)

with tf.Session() as sess:
    embedder = Embedder()
    embeddings_init_ = embedder.load('talkchain-50', task.get_words())
    sess.run(tf.global_variables_initializer(), feed_dict={embeddings_init: embeddings_init_})
    tr_loss = {}

    for j in itertools.count():
        sentences_ids_ = task.train_batch()
        tr_loss[j], _ = sess.run([loss, minimize], {sentences_ids: sentences_ids_})
        debug(j, tr_loss, tr_loss, debug_steps, 'tr')
