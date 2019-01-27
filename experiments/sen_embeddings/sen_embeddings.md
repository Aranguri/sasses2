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

the idea is to find the manifold of sentences/concepts.

Inspect how sentences are recovered. Where it fails? How does the recovering quality improve as we increase the size of the hidden state.

for test time, we need to avoid passing the value of the correct sentence as input in the next step, but we pass what it said as the next input.

If this works, it would be interesting if we have a manifold in space where changing a little bit the position of the concept point changes a little bit the words we get as a result (try this.)

Steps:
0) Go from 1-4 until we reach an acceptable performance. When that happens skip to (5)
1) add dropout (if dev >> tr)
2) output_mode: add a mlp instead of a projection (if tr >> 0)
3) increase training size (if dev >> tr)
3.5) increase embeddings size (if tr >> 0)
4) what other things can I do to increase performance? (eg vary hyperparams)
5) define a simple similarity function between concepts
6) given one initial concept, produce the next going to the concept that is most similar

Logs
## 1
Around 7000 iterations, it reaches .07 training loss
## 2
it seems weird to me that it gets that good performance at recovering the sentence. It was wrong. The input to the rnn was the expected output. hehe

Other
The relative change between GPU and cpu is 100x
next time: write progress of train and dev set into file.

duplex --> talking to customers
may coterm in philsophy -> thinks of philsoophy as code interesting topic
looking for research --> phds
did a lot of ai classes --> looking for a job
we could go for meal
aneesh pappu + profesor
backpacking + no saying his stanf undergrad title + writing in parks + plans to do that every year. in some weeks to balley indonesia with friends

## Running something

19514.pts-4.jacob (1e-3)

### 19587.pts-4.jacob (1e-4)
Train: 2.1309609413146973 Dev: 2.3221688270568848
This gives hope for the structure and the general meaning is pretty good. (eg, it wrote I instead of he, and it wrote puncuation marks whenever they were in the input sentence.) So it seems it gets right the categories but not the specific meanings yet. It's inexact but in the correct direction.
> `` he ca n 't get me .                               
 `` i 's n 't be into ,                               n

### 19713.pts-4.jacob (1e-5)
83k iterations (we can't directly compare this to the other exp)
Train: 2.28 Dev: 2.55

It's particularly good with punctuation marks. In this case, wherever there was one punctuation mark in the input, it wrote some (potentially different) puncuation mark. (It wrote some more, though). This model doesn't seem as good gramatically as the one below, though we can't know if it's because of not training that much or because of the learning rate or because of proj+attn.
> who draws the water , or he who empties it ? '
 ` why , certainly he who draws the water ! '
 ` you hear ? '
 said the jackal , turning to the sheep .

> had the little , and the had had , ,
 ` i , ' , will will , king , '
 ` i are , '
 asked the king , and the the other .n

400k iterations
Train around 2 (variable between 1.2 and 3.2). Dev: 2.3

### 20339.pts-4.jacob (1e-4 and proj+attn)
206k iterations
dev set: 2.24
tr set: (avg) 1.8

1.1M iterations
not that many changes in the tr set. Dev set got worse (around 2.6)

#### Quality
##### Trainign set
it learend to capture statistical patterns in how words are associated.
> are all the road - of the island , and the first of)


Here it perfectly recovered the structure of the sentence. Interestingly, it wrote man instead of gentleman. That's an advantage of attending a matrix with the WE, I think. The advantage I refer to is that if we wanted to say gentleman with a vector that was gentleman + noise, then we may end up saying man but not chair.
>`` how soundly he sleeps ! ' '
whispered the old gentleman .                         

`` i can they shall ! ' '
cried the old man .              

it captures some entities so far. It's also interesting that it captures the entities exactly in the right place
>the king was unwilling to risk his third son on such an errand , but he begged so long that his father had at last to consent .
` king was very to be the own son , the a purpose , and he was to much that he father would no once told ask .
`
### Conclusions
The model is underfitting the training data. Possible things happening:
* not enough iterations
* small learning rate
* embedding size too small
* max_seq len too large

theoretically, the problem is trivial if the hidden size is greater than or equal to the embedding size * seq_length. You just concatenate all the embeddings. The problem is that it's not clear how a LSTM could concatenate things. This is just NMT dude.

# Steps
* add max norm
* fix accuracy
* check how is the output of the guys
* see how big is my training set compared to theirs
* use bigger embedding_size?
