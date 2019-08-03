import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, datasets, iterators, optimizers
from chainer.training import extensions
import numpy as np
import sys
import codecs

# RNNクラス
class Generate_RNN(chainer.Chain):

    def __init__(self, n_words, nodes):
        super(Generate_RNN, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_words, n_words)
            self.l1 = L.LSTM(n_words, nodes)
            self.l2 = L.LSTM(nodes, nodes)
            self.l3 = L.Linear(nodes, n_words)

    # 内部ステータスのリセット
    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def __call__(self, x):
        h0 = self.embed(x)
        h1 = self.l1(h0)
        h2 = self.l2(h1)
        y = self.l3(h2)

        return y

s = codecs.open("")