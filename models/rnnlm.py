import cupy as np

from ..module.layers import *


class RNNLM :
    def __init__(self, vocab_size, word_vec_size, hidden_size, length, rnn='rnn') :
        W_emb = np.random.randn(vocab_size, word_vec_size) * (2. / np.sqrt(vocab_size + word_vec_size))

        H = hidden_size * 4 if rnn == 'lstm' else hidden_size
        Wh = np.random.randn(hidden_size, H) * (2. / np.sqrt(hidden_size + H))
        Wx = np.random.randn(word_vec_size, H) * (2. / np.sqrt(word_vec_size + H))
        b_rnn = np.zeros(H)

        W_lin = np.random.randn(hidden_size, vocab_size) * (2. / np.sqrt(hidden_size + vocab_size))
        b_lin = np.zeros(vocab_size)

        self.vocab_size = vocab_size
        self.word_vec_size = word_vec_size
        self.length = length
        self.hidden_size = hidden_size
        self.batch_size = None

        self.emb = [Embedding(W_emb) for _ in range(length)]
        if rnn == 'lstm' :
            self.rnn = LSTM(Wh, Wx, b_rnn)
        elif rnn == 'rnn' :
            self.rnn = RNN(Wh, Wx, b_rnn)
        self.lin = [Linear(W_lin, b_lin) for _ in range(length)]
        self.loss = [Softmax(vocab_size) for _ in range(length)]

        self.layers = self.emb + [self.rnn] + self.lin + self.loss

        self.params, self.grads = [], []
        self.params = [W_emb,
                       Wh, Wx, b_rnn,
                       W_lin, b_lin]
        self.grads = [np.zeros_like(p) for p in self.params]
        self.rnn_layers = [self.rnn]

    def forward(self, xs, ys, is_train=True) :
        batch_size = len(ys)
        self.batch_size = batch_size
        embs = np.zeros((batch_size, self.length, self.word_vec_size), dtype='f')
        for t, layer in enumerate(self.emb) :
            embs[:, t, :] = layer.forward(xs[:, t])
        # |embs| = (batch_size, length, word_vec_size)
        hs = self.rnn.forward(embs)
        # |hs| = (batch_size, length, hidden_size)
        zs = np.zeros((batch_size, self.length, self.vocab_size))
        for t, layer in enumerate(self.lin) :
            zs[:, t, :] = layer.forward(hs[:, t, :])
        # |zs| = (batch_size, length, vocab_size)
        loss = 0.
        for t, layer in enumerate(self.loss) :
            loss += layer.forward(zs[:, t, :], ys[:, t])
        loss /= self.length
        return loss

    def backward(self, dout=1.) :
        grads = [0., 0., 0., 0., 0., 0.]

        dzs = np.zeros((self.batch_size, self.length, self.vocab_size), dtype='f')
        for t, layer in enumerate(self.loss) :
            dzs[:, t, :] = layer.backward(dout)

        dhs = np.zeros((self.batch_size, self.length, self.hidden_size), dtype='f')
        for t, layer in enumerate(self.lin) :
            dhs[:, t, :] = layer.backward(dzs[:, t, :])
            for i in range(2) :
                grads[4 + i] += layer.grads[i]

        dembs = self.rnn.backward(dhs)
        for i in range(3) :
            grads[1 + i] = self.rnn.grads[i]

        dxs = np.zeros((self.batch_size, self.length))
        for t, layer in enumerate(self.emb) :
            layer.backward(dembs[:, t, :])
            grads[0] += layer.grads[0]

        for i in range(6) :
            self.grads[i][...] = grads[i]

        return None