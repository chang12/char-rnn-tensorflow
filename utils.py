import codecs
import collections
import os
import random
from six.moves import cPickle

import numpy as np


class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, encoding="utf-8"):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding
        self.chars = None
        self.vocab_size = None
        self.vocab = None
        self.tensor = None
        self.num_batches = None
        self.x_batches = None
        self.y_batches = None

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.pre_process(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()

    def pre_process(self, input_file, vocab_file, tensor_file):
        # self.tensor 는 index 의 np.array
        # frequency 가 높은 character 에 작은 index 값이 배정되도록 sorted

        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            data = f.read()
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)
        self.tensor = np.array(list(map(self.vocab.get, data)))
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length))

    def create_batches(self):
        # tensor.size = 200 / batch_size = 5 / seq_length = 10 -> num_batches = 4
        # tensor.reshape(batch_size, -1) -> (5, 40)
        # len(x_batches) = num_batches
        # x_batches 의 각 element 들은 2d nd.array 가 되고, 각 row 가 sequence 가 된다. sequence 가 batch_size 만큼 있는 nd.array

        self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length))
        if self.num_batches == 0:  # When the data (tensor) is too small, let's give them a better error message
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        x_data = self.tensor
        y_data = np.copy(self.tensor)
        y_data[:-1] = x_data[1:]
        y_data[-1] = x_data[0]
        self.x_batches = np.split(x_data.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(y_data.reshape(self.batch_size, -1), self.num_batches, 1)

    def next_batch(self):
        random_pointer = random.randint(0, self.num_batches)
        x, y = self.x_batches[random_pointer], self.y_batches[random_pointer]
        return x, y
