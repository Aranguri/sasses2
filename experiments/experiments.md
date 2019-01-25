# Talk-chain
## Description
We want something with whom we can have a conversation. One way of answering in a conversation is by talking about related topics to what the other person talked about. We have a space of concepts. And we encode the current topic of the conversation in a concept. Then, we look for similar concepts in the space of concepts. And based on the concept we create an answer sentence. We would a need a rnn-autoencoder.

The problem is how we are going to train it. Oh. We can do

sen_1
sen_2
...
sen_n

We then use a RNNAutoencoder to create a hidden vector for every sentence. And we store those vectors into memory.

Then, given those vectors, the MLP has to select a vector from the memory and reconstruct it. The selection of the vector is supervised, because we know what's the next sentence in the in the paragraph.

We can experiment and think about how big the hidden vector should be to efficiently store the sentence.

Let's first experiment with the rnnautoencoder

### rnnautoencoder
We need to create a representation for a sentence. the first thing we are going to try is a rnn that encodes the sentence into one hidden vector. And then we need to recover the sentence from that vector.

What's the interface to WE?
W: a matrix with word embeddings in its columns.
u: a word represented in one-hot encoding.
Wu: a word represented in word-embeddings
W^{-1} (Wu): how would this work?

W is a embeddings_size x vocab_size matrix. Eg, 64 x 1000.

The question is how we tie f ang g with f: one-hot -> WE and g: WE -> one-hot. The initial problem with this is that say we have a matrix A that implements f.

We can start without the input and output tied. 

## Things
We can try both word and char-level.
We need to experiment in the different ways to implement the rnnautoencoder.
can we express the wordembeddings as a matrix A and then take its pseudoinverse to go from WE to one-hot? We can mantain the expression as UEV^T to make it more efficient. the problem seems to be that A is huge. A dense multiplication would take 128 * 500 * 100k = 5000M multiplicatoins (batch_size: 128, word embeddings: 500.) however, there could be an easier way of doing this matrix multiplication because it consists in one rotation, scaling and another rotation. we'll see. {understand gpus and how big the matrices can be.}
it would be interesting to try encoding in word level and decoding in char level. or vice versa.


# Internet
Is there a way to download (part of :) the internet without images. So as to give an agent freedom to choose (almost) any link but without having to wait for the request.
