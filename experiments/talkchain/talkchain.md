Inspect how sentences are recovered. Where it fails? How does the recovering quality improve as we increase the size of the hidden state.

for test time, we need to avoid passing the value of the correct sentence as input in the next step, but we pass what it said as the next input.

Steps:
+ add dev loss
* check loss and quality of output sentences with different methods (specifically check the output projection)

Logs
## 1
Around 7000 iterations, it reaches .07 training loss
## 2
it seems weird to me that it gets that good performance at recovering the sentence. It was wrong. The input to the rnn was the expected output. hehe 

Other
The relative change between GPU and cpu is 100x
