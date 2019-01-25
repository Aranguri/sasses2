import os
import pickle
import numpy as np

embeddings_path = '../../datasets/glove/glove.6B.50d.pickle'

class Embedder:
    def __init__(self):
        pass

    def load(self, name, words):
        # Future: make it possible to use different embedding size as an argument
        file_path = f'../../datasets/embeddings/{name}.pickle'
        if not os.path.isfile(file_path):
            self.generate(file_path, words)
        #check for file if itsn't, geenrate it
        with open(file_path, 'rb') as handle:
            return pickle.load(handle)

    def generate(self, file_path, words):
        with open(embeddings_path, 'rb') as handle:
            self.weights = pickle.load(handle)

        # note: the first vector in the word embeddings is for the empty character.
        #  Thus, we assign a random vector for the empty character. note that
        #  in DictTask.load_from_file we took into account this and added one to the indexes.
        embeddings = np.array([[self.rand_weight()]])
        embeddings = np.array([self.embed(word) for word in words])

        with open(file_path, 'wb') as handle:
            pickle.dump(embeddings, handle)

    def embed(self, word):
        if word not in self.weights.keys():
            self.weights[word] = self.rand_weight()
        return self.weights[word]

    def rand_weight(self):
        dims = len(self.weights['a'])
        # note: .7 is the expected standard deviation of the vectors in glove
        return np.random.randn(dims) * .7
