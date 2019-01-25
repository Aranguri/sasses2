import sys
sys.path.append('../')
from utils.util import *
from keras.preprocessing.sequence import pad_sequences
import os
import pickle

class SenTask:
    def __init__(self, batch_size, char_limit, seq_length_limit, exp_name):
        self.batch_size = batch_size
        file_path = f'../../datasets/tasks/{exp_name}.pickle'
        if not os.path.isfile(file_path):
            self.generate(char_limit, seq_length_limit, file_path)
        else:
            with open(file_path, 'rb') as handle:
                self.batches, self.vocab_size, self.word_to_i, self.i_to_word = pickle.load(handle)

        max_len = max([np.shape(b)[1] for b in self.batches])
        print(max_len)
        exit()
        self.train = self.batches[:-10]
        self.dev = self.batches[-10:]
        self.t_i = 0
        self.d_i = 0

    def generate(self, char_limit, seq_length_limit, file_path):
        text = open('../../datasets/childrens_book/data/cbt_train.txt').read()[:char_limit]
        words = clean_text(text)

        self.vocab_size, self.word_to_i, self.i_to_word, data = tokenize_words_simple(words)
        splits = np.where(data == self.word_to_i['.'])[0] + 1
        data = np.split(data, splits)
        self.batches = []
        for i in range(len(data) // self.batch_size):
            batch = data[i * self.batch_size:(i + 1) * self.batch_size]
            batch = pad_sequences(batch, max_len=seq_length_limit, padding='post')
            self.batches.append(batch)

        to_store = [self.batches, self.vocab_size, self.word_to_i, self.i_to_word]
        with open(file_path, 'wb') as handle:
            pickle.dump(to_store, handle)

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
