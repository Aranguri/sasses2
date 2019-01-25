import sys
sys.path.append('../')
from util import *
from keras.preprocessing.sequence import pad_sequences

class SenTask:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        text = open('../../datasets/pg.txt').read()[:1000000]
        words = clean_text(text)
        # print(words)

        self.vocab_size, self.word_to_i, self.i_to_word, data = tokenize_words_simple(words)
        splits = np.where(data == self.word_to_i['.'])[0] + 1
        data = np.split(data, splits)
        batches = []
        for i in range(len(data) // batch_size):
            batch = data[i * batch_size:(i + 1) * batch_size]
            batch = pad_sequences(batch, padding='post')
            batches.append(batch)

        self.train = batches[:-10]
        self.dev = batches[-10:]
        self.t_i = 0
        self.d_i = 0

    def train_batch(self):
        self.t_i = 0 if self.t_i + 1 == len(self.train) else self.t_i + 1
        return self.train[self.t_i]

    def dev_batch(self):
        self.d_i = 0 if self.d_i + 1 == len(self.dev) else self.d_i + 1
        return self.dev[self.d_i]

    def get_lengths(self):
        return self.vocab_size

    def get_words(self):
        return self.word_to_i.keys()

    def ixs_to_words(self, ixs):
        return self.i_to_word[ixs]
