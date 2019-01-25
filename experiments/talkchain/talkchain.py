import sys
sys.path.append('../../')
from tasks.sentences.sen_task import SenTask
from utils.embedder.embedder import Embedder
import itertools
import tensorflow as tf
from utils.util import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
args = parser.parse_args()

exp_name = 'exp:talkchain,task:cbt,embed:50,v:2'
batch_size = 128
hidden_size = 512
learning_rate = args.lr
debug_steps = 100
embeddings_size = 50 # It's fixed from glove
char_limit = -1#25000000
seq_length_limit = 40
exp_name = f'{exp_name},limit:{char_limit},batch_size:{batch_size},seq_length:{seq_length_limit}'
running_GPU = True

if running_GPU:
    LSTM = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell
else:
    LSTM = tf.contrib.rnn.LSTMBlockCell
task = SenTask(batch_size, char_limit, seq_length_limit, exp_name)
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
shifted_inputs = tf.concat((start_stacked, inputs[:, :-1]), axis=1)

rnn_fw = LSTM(hidden_size)
rnn_bw = LSTM(hidden_size)
_, final_state = tf.nn.dynamic_rnn(rnn_fw, inputs, dtype=tf.float32)
outputs, _ = tf.nn.dynamic_rnn(rnn_fw, shifted_inputs, dtype=tf.float32)
# Other options for the following function: (1) keep generating a prob. dist. directly, but use something more complex, like a mlp. (2) produce a vector in embedding space and compute the similarity over all the memories (aka attention.) Then we have a probability distribution.
outputs = tf.einsum('ijk,kl->ijl', outputs, projection)

loss = tf.losses.softmax_cross_entropy(sentences_hot, outputs)
optimizer = tf.train.AdamOptimizer(learning_rate)
minimize = optimizer.minimize(loss)

def debug_output(answer, output):
    ixs = np.argmax(output[0], axis=1)
    original = ' '.join(task.ixs_to_words(answer[0]))
    recovered = ' '.join(task.ixs_to_words(ixs))
    print(original + '\n' + recovered + 'n\n-----------')

with tf.Session() as sess:
    embedder = Embedder()
    embeddings_init_ = embedder.load(exp_name, task.get_words())
    sess.run(tf.global_variables_initializer(), feed_dict={embeddings_init: embeddings_init_})

    tr_loss, dev_loss = {}, {}
    for j in itertools.count():
        sentences_ids_ = task.train_batch()
        outputs_, tr_loss[j], _ = sess.run([outputs, loss, minimize], {sentences_ids: sentences_ids_})

        if j % debug_steps == 0:
            debug_output(sentences_ids_, outputs_)
            sentences_ids_ = task.dev_batch()
            outputs_, dev_loss[j] = sess.run([outputs, loss], {sentences_ids: sentences_ids_})
            debug_output(sentences_ids_, outputs_)

        plot_mode = 'none' if running_GPU else 'tr'
        debug(j, tr_loss, dev_loss, debug_steps, plot_mode)
