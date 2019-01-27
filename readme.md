# General
## What's this?
Memory-augmented neural networks are interesting. We want to see how we can make them more useful for general tasks (as opposed to shallow ones.) Thus, we want to measure the performance on general tasks of the current models. And we want to see what features we can add to make this performance go up. Goal: in the end, having a little framework. But remember, start very, very simple!

## Instructions for running
The interesting things are in the experiments folder. After cloning this repo, you can run cd experiments/talkchain/ and python talkchain.py to get the chain-of-thoughts experiment running. You need python 3.6 and the datasets, that you can download from https://drive.google.com/file/d/1RsuDu5i70rIrh-jzoTDaHQWD0Ie4V_A7&export=download. This folder should be located in the root of the project. Have fun :)

## API
The idea is for runner and task to be functions that can be run independently of the actual task and mann we are using.

# Done things and things maydo
## Experiments
### Similarity functions
What are the pros and cons of the different similarity functions? Is there any similarity function that dominates the others? If we use both additive and multiplicative interactions, do we enable better performance? Not only code, but theoretically think about this.
Some similarity functions
* hadamard
* cosine distance
* inner product

### Sentence embeddings
Look at the manifold created by the approaches below. How is it to go through this manifold?

Some approaches
* english-to-english NMT
* mean or median of the WEs
* LSTM with constrained gradient updates (so as to simulate n LSTMs.): this could be useful to have different detectors of features where every one specializes in doing one feature extraction.  

### Talkchain
Say we define some interesting encoding for sentences. We'd like this encoding to put near the sentences that are similar. First, we encode lots of sentences, and we store that as our long-term memory. Then, we write natural language by starting in one point in this manifold and go to the points that are near. (We could have a short-term memory that decides the direction. We could have momentum.)

### Compositionality
For a set of _similar_ memories, we find one representative. Then, we express the set of similar memories as the representative plus a bias.

### Others
* What goes from short to long-term memory?
* Flags (something like emotions): time, importance
* Is there a way to download part of the internet (only text.) So as to give an agent freedom to choose almost any link but without having to wait for the request. (Otherwise standard deep learning probably won't work, because we wouldn't have enough training cases.)
* Recursive hopfield net
* Analogies

## Manns
### DMN
### NTM

## Tasks
### ATP
### Definitions from dictionaries
Say we have two different dictionaries with the same words but different definitions. We let the learning model memorize the <word, definition> pairs from the first dictionary. Then, we give the pair <word, definition'> from the second dictionary (ie, one word from the first dictionary, defined in another way.) We ask the learning model to output word given its memories and definition'.

### Others
* more standard ones

## Learning
### Papers
Finish papers

### Misc
Understand unixpickle's github
does crossentropy loss care about the order of the arguments.

## Others
### Maydo
pass all the info into here.
### Resources
scp -r datasets aranguri@sc.stanford.edu:sasses


# MANNs
## Code
### Model
We want to have this model, where we persist. Where we can try different similarity functions and do all sorts of experiments. Where we can try generalization tasks. That model could be a DMN. A model where we can test how it works to connect to the internet. Also, a model where we can test curiosity. And do the things that make me the happiest, as experiments of memnns.

#### Specifics
I imagine it as follows. We have one library for tasks, one library for memnns, and then we experiment by including task/s and memnn/s. In experiments, we could have one folder for each thing we want to try

### Representations and similarity
#### Variable-length sentences
Say we have two sentences of different length. One way to compare them is to represent both sentences into one hidden vector (we can use a RNN for this.) But this is limited: no matter how long the input sentences are, we encode them into one fixed representation.

The problem with doing otherwise is that generally, starting with two sentences of different lengths implies ending with two representations for the sentences that are of different length.

We can use ten rnns instead of only one. We can interpret each rnn as computing one function. Question: is a RNN with 1280 hidden units equally powerful to 10 RNNs with 128 units?
{}

### Compositionality
#### Modules
Say we have n mlps, and we connect them all together

## Notes
### Essays
* Introduction to memory networks
* Long-term memories

### Papers
* Ask Me Anything. Dynamic Memory Networks for Natural Language Processing
* Differentiable Neural Computer {incomplete}
* Memory Networks {incomplete}
* Scaling Memory-Augmented Neural Networks with Sparse Reads and Writes {incomplete}
* Neural Turing Machine {on sheet of paper}
* MAC Network {on sheet of paper}
* Attention and Augmented Recurrent Neural Networks


## Future things
create a roadmap
what if a nn is connected to the internet?
are computers taking advantage of good assumptions we can make on the type of data humans have? does this even make sense?
is there a way of using the kernel trick? we could think the memory at time t as a linear combination of the input and the memory at previous timesteps
We need a long-term memory.
* we can't read all the memories at once. So we need to try with something like a tree (hierarchical structure), a graph (relational structure), hard attention, or something I call recursive hopfield net. [3]
* we need to know what we are going to store.
* we need to model long-term dependnces, but backpropagation through large amounts of time doesn't work that well. We need to explore this more.
* we can connect it to the internet
* we can let the nn communicate with each other
* we should also have a short-term memory, and decide what goes from there to the long-term (we can use: the less you attend something, the more it disappears from your short-term memory. Also, the more you attend something, the higher are the chances of saving it into your memory.)
Task: given a description in wikimedia, it has to generate the label (or vice versa.)

is the softargmax also a gate?
next steps
* look for a dataset where it makes sense to have a huge memory
* fix memory not changing in the same run

* Problems: I give you eight real numbers. You have only four slots to store real numbers. How precise can your answer be? Can you recover the full four numbers?
* the output of the RNN_{MG} has memory_size. What if it only has size 1. ie, what if we write one memory instead of the whole thing.
* {consider that in my implementation each input receives only one iteration of processing through the unit. it would be interesting to have as many processing units as the neural nets thinks is good}

* https://github.com/unixpickle/sgdstore

lagrange multipliers

### Papers
* Neural RAM: https://arxiv.org/pdf/1511.06392.pdf
* Neural gpus: https://arxiv.org/abs/1511.08228
* MAC: https://arxiv.org/pdf/1803.03067.pdf
* RL-NTM: https://arxiv.org/pdf/1505.00521.pdf
* End to end memory networks: https://arxiv.org/pdf/1503.08895.pdf
* One-shot learning: https://arxiv.org/pdf/1605.06065v1.pdf
