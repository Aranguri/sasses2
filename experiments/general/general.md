We want to model P(y_t|x_t, m_t; W)

All the functions belw are parameterized by a portion of W.
1) x'_t = transform(x_t)
2) y'_t = f(x'_t, m)
3) y_t = transform(y'_t)
4) m_{t+1} = g(x'_t, y'_t, m_t)

There is a more generalized setting where instead of receiving an input and answering at every timestep, we can skip steps.

The step 1 is related to what we are looking at in long-term. It also depends on step 2, for the representation depends on the usage.
The step 2 is almost all of the problem. Let's think about different ways to implement it to get a flavor.
	a) Table-based algorithm. m is a table with entries <input, output>. Then, given an input x, we just look up for the output y. Cons: we need a lot of storage and we don't generalize.
	b) Key-value memories. m is a table with entries <input, output>. Given x, we retrieve the output whose input is the closest to x. Note: this model is good for an environment where we have lots of different inputs but few different outputs. For instance, if we want to model the function f(x) = round(x) * 10 (eg f(4.6) = 50, f(3.14) = 30) for the first hundred naturals, then we just need the entries [<1, 10>, <2, 20>, <3, 30>, ... <100, 1000>]. This will work because when we compute the distance from x to every input in the memory, we are implicitly computing the round function.
	\hat b) Combination of memories in series. What if we have a basis for the memories.
	\tilde b) placeholders.
	r) multiple steps of reasoning. divide and conquer.
	z) the more general setting here seems to be a rnn with external memory with ACT. We need the ACT because different inputs require different amount of memories retrieved and processing steps.



is softargmax always used to just get the argmax in a soft way or sometimes ends up merging two or three memories/input vectors in a useful way
Something to consider is how we store some knowledge on the memory and other knowledge inside the function f (and potentially transform.) We also want to make the distinction between the neural network and memory because: (a) we don't want to read/modify all the memories at once, (b) we may want to create something like WE for memory, (c) we don't want to modify.

so the idea is to have an RNN with an oracle that is the memory.

We want a general purpose cell where instead of just having the short-term memory as a LSTM has, we also have the long-term memory.
