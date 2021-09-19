import pickle

import cupy as np

from ..module.layers import *


class LSTMLM :
    def __init__(self,
                 vocab_size,
                 word_vec_size,
                 hidden_size,
                 dropout_p=.5,
                 var_dropout_p=0.,
                 emb_dropout_p=0.,
                 bidirectional=False) :

        W_emb = np.random.randn(vocab_size, word_vec_size).astype('f') * (2. / np.sqrt(vocab_size + word_vec_size))

        W_h1 = np.random.randn(hidden_size, 4 * hidden_size).astype('f') * (2. / np.sqrt(hidden_size + hidden_size))
        W_x1 = np.random.randn(word_vec_size, 4 * hidden_size).astype('f') * (2. / np.sqrt(word_vec_size + hidden_size))
        b_lstm1 = np.zeros(4 * hidden_size).astype('f')

        W_h2 = np.random.randn(hidden_size, 4 * hidden_size).astype('f') * (2. / np.sqrt(hidden_size + hidden_size))
        W_x2 = np.random.randn(hidden_size, 4 * hidden_size).astype('f') * (2. / np.sqrt(hidden_size + hidden_size))
        b_lstm2 = np.zeros(4 * hidden_size).astype('f')

        b_lin = np.zeros(vocab_size).astype('f')

        self.layers = [Embedding(W_emb),
                       Dropout(dropout_p, recur=True),
                       BiLSTM(W_h1, W_x1, b_lstm1, var_dropout_p) if bidirectional else LSTM(W_h1, W_x1, b_lstm1,
                                                                                             var_dropout_p),
                       Dropout(dropout_p, recur=True),
                       BiLSTM(W_h2, W_x2, b_lstm2, var_dropout_p) if bidirectional else LSTM(W_h2, W_x2, b_lstm2,
                                                                                             var_dropout_p),
                       Dropout(dropout_p, recur=True),
                       Affine(W_emb.T, b_lin),
                       Softmax(vocab_size)]

        self.rnn_layers = [self.layers[2], self.layers[4]]

        if emb_dropout_p > 0 :
            self.layers.insert(0, EmbeddingDropout(emb_dropout_p))

        self.params, self.grads = [], []
        for layer in self.layers :
            self.params += layer.params
            self.grads += layer.grads

    def load_params(self, load_fn) :
        with open(load_fn, 'rb') as f :
            params = pickle.load(f)

        for i, p in enumerate(params['params']) :
            self.params[i][...] = p

    def reset_state(self) :
        for layer in self.rnn_layers :
            layer.reset_state()

    def forward(self, xs, ys, is_train=True) :
        for layer in self.layers[:-1] :
            xs = layer.forward(xs, is_train=is_train)
        loss = self.layers[-1].forward(xs, ys)
        return loss

    def backward(self, dout=1.) :
        for layer in reversed(self.layers) :
            dout = layer.backward(dout)
        return None