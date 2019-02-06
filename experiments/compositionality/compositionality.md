[1] Though it would be interesting to constrain the representatives to be one memory. For humans, it seems we can only
[2] We can also let the memories be a linear combination of representatives, instead of just using one.

Shared language: question and text
Meta normalize
Is there cases where softmax doesn't work


Say we have the hidden state of the LSTM. And we have n templates

checkout other branch

add loss

hidden_state =                   (batch_size x h)
templates = {v_1, v_2, ..., v_n} (batch_size x num_templates x h)
similarities = tf.einsum('ijk, ik -> ij', templates, h)
similarities = tf.nn.softmax(similarities)
new_hidden = tf.einsum('ijk, ij -> ik', templates, similarities)


Conda stuff:
source activate tf...
~/miniconda3/envs/bidaf/bin/pip install PACKAGES
URL for r0.11: https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0-cp35-cp35m-linux_x86_64.whl



