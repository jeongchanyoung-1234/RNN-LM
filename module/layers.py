import cupy as np
import cupyx

from functions import softmax, sigmoid


class MatMul :
    def __init__(self, W) :
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.cache = None

    def forward(self, x) :
        self.cache = x
        W, = self.params
        out = np.matmul(x, W)
        return out

    def backward(self, dout) :
        x = self.cache
        W, = self.params
        dW = np.matmul(x.T, dout)
        dx = np.matmul(dout, W.T)
        self.grads[0][...] = dW
        return dx


class _RNN :
    def __init__(self, W_h, W_x, b) :
        # |W_h| = (hidden_size, hidden_Size)
        # |W_b| = (vocab_size, hidden_size)
        # |b| = (hidden_size, )
        self.W_h = MatMul(W_h)
        self.W_x = MatMul(W_x)

        self.params = self.W_h.params + self.W_x.params + [b]
        self.grads = self.W_h.grads + self.W_x.grads + [np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_t_1, is_train=True) :
        W_h, W_x, b = self.params
        batch_size = x.shape[0]
        hidden_size = W_h.shape[0]
        if h_t_1 is None :
            h_t_1 = np.zeros((batch_size, hidden_size))
        h_t = np.tanh(self.W_h.forward(h_t_1) + self.W_x.forward(x) + b)
        self.cache = h_t
        return h_t

    def backward(self, dh_t) :
        h_t = self.cache
        dh_t = dh_t * (1 - h_t ** 2)
        db = dh_t.sum(axis=0)
        dh_t_1 = self.W_h.backward(dh_t)
        dx = self.W_x.backward(dh_t)
        self.grads[0][...] = self.W_h.grads[0][...]
        self.grads[1][...] = self.W_x.grads[0][...]
        self.grads[2][...] = db
        return dx, dh_t_1


class RNN :
    def __init__(self, Wh, Wx, b) :
        self.params, self.grads = [Wh, Wx, b], [np.zeros_like(Wh), np.zeros_like(Wx), np.zeros_like(b)]
        self.layers = None
        self.h_t = None

        self.batch_size, self.length, self.hidden_size = None, None, None

    def reset_state(self) :
        self.h_t = None

    def forward(self, x, is_train=True) :
        Wh, Wx, b = self.params
        # |x| = (batch_size, length, vocab_size)
        self.batch_size = x.shape[0]
        self.length = x.shape[1]
        self.word_vec_size = x.shape[2]
        self.hidden_size = Wh.shape[0]

        hs = np.zeros((self.batch_size, self.length, self.hidden_size), dtype='f')
        h_t = self.h_t

        self.layers = []
        for t in range(self.length) :
            rnn = _RNN(Wh, Wx, b)
            h_t = rnn.forward(x[:, t, :], h_t)
            hs[:, t, :] = h_t
            self.layers.append(rnn)

        self.h_t = h_t
        # |hs| = (bs, t, hs)
        return hs

    def backward(self, dh_ts) :
        dxs = np.zeros((self.batch_size, self.length, self.word_vec_size))
        grads = [0., 0., 0.]
        t = len(self.layers) - 1
        for rnn in reversed(self.layers) :
            if t == len(self.layers) - 1 :
                dx, dh_t_1 = rnn.backward(dh_ts[:, t, :])
            else :
                dx, dh_t_1 = rnn.backward(dh_ts[:, t, :] + dh_t_1)
            dxs[:, t, :] = dx
            for i, grad in enumerate(rnn.grads) :
                grads[i] += grad
            t -= 1
        for i, grad in enumerate(grads) :
            self.grads[i][...] = grad

        return dxs


class Linear :
    def __init__(self, W, b) :
        self.matmul = MatMul(W)
        self.params = self.matmul.params + [b]
        self.grads = self.matmul.grads + [np.zeros_like(b)]

    def forward(self, x, is_train=True) :
        b = self.params[1]
        out = self.matmul.forward(x) + b
        return out

    def backward(self, dout) :
        db = dout.sum(axis=0)
        dx = self.matmul.backward(dout)

        self.grads[0][...] = self.matmul.grads[0][...]
        self.grads[1][...] = db
        return dx

# class Embedding:
#     def __init__(self, W):
#         self.params = [W]
#         self.grads = [np.zeros_like(W)]
#         self.idx = None
#
#     def forward(self, idx, is_train=True):
#         W, = self.params
#         self.idx = idx
#         out = W[idx]
#         return out
#
#     def backward(self, dout):
#         dW, = self.grads
#         dW[...] = 0.
#         cupyx.scatter_add(dW, self.idx, dout)
#         return None

class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx, is_train=True):
        self.shape = idx.shape
        self.dim = idx.ndim
        if self.dim == 2:
          idx = idx.flatten()
        W, = self.params
        self.idx = idx
        out = W[idx]
        if self.dim == 2:
          out = out.reshape(self.shape[0], self.shape[1], -1)
          # |out| = (bs, t, hs)
        return out

    def backward(self, dout):
        if self.dim == 2:
          dout = dout.reshape(self.shape[0] * self.shape[1], -1)
        dW, = self.grads
        dW[...] = 0
        cupyx.scatter_add(dW, self.idx, dout)
        return None


class _LSTM :
    def __init__(self, W_h, W_x, b) :
        # |W_h| = (hs, hs * 4)
        # |W_x| = (ws, hs * 4)
        # |b| = (hs * 4)

        self.params = [W_h, W_x, b]
        self.grads = [np.zeros_like(p) for p in self.params]
        self.cache = None

    def forward(self, x, h_t_1, c_t_1) :
        W_h, W_x, b = self.params
        z = np.dot(h_t_1, W_h) + np.dot(x, W_x) + b  # repeat
        # |z| = (bs, hs * 4)
        hidden_size = h_t_1.shape[1]

        f = sigmoid(z[:, :hidden_size])
        g = np.tanh(z[:, hidden_size :2 * hidden_size])
        i = sigmoid(z[:, 2 * hidden_size :3 * hidden_size])
        o = sigmoid(z[:, 3 * hidden_size :])

        c_t = c_t_1 * f + g * i
        c_t_tanh = np.tanh(c_t)
        h_t = o * c_t_tanh

        self.cache = f, g, i, o, x, h_t_1, c_t_1, c_t_tanh
        return h_t, c_t

    def backward(self, dh_t, dc_t) :
        W_h, W_x, b = self.params
        f, g, i, o, x, h_t_1, c_t_1, c_t_tanh = self.cache

        do = c_t_tanh * dh_t
        do = do * o * (1. - o)

        dc_t = dc_t + (dh_t * o) * (1. - c_t_tanh ** 2)
        dc_t_1 = f * dc_t

        df = c_t_1 * dc_t
        df = df * f * (1. - f)

        dg = i * dc_t
        dg = dg * (1. - g ** 2)

        di = g * dc_t
        di = di * i * (1. - i)

        dz = np.hstack([df, dg, di, do])
        # |dz| = (bs, 4 * hs)
        db = dz.sum(axis=0)

        dh_t_1 = np.matmul(dz, W_h.T)
        dW_h = np.matmul(h_t_1.T, dz)
        dx_t = np.matmul(dz, W_x.T)
        dW_x = np.matmul(x.T, dz)

        self.grads[0][...] = dW_h
        self.grads[1][...] = dW_x
        self.grads[2][...] = db

        return dx_t, dh_t_1, dc_t_1


class LSTM :
    def __init__(self, Wh, Wx, b, var_dropout_p=0.) :
        self.params = [Wh, Wx, b]
        self.grads = [np.zeros_like(Wh), np.zeros_like(Wx), np.zeros_like(b)]

        self.hidden_size = Wh.shape[0]
        self.length = None
        self.input_size = Wx.shape[0]

        self.h_t, self.c_t = None, None
        self.layers = None

        self.var_dropout_p = var_dropout_p
        self.mask0 = None
        self.mask1 = None

    def forward(self, xs, is_train=True) :
        # |xs| = (bs, t, is)
        Wh, Wx, b = self.params
        batch_size = xs.shape[0]
        self.length = xs.shape[1]

        self.layers = []
        hs = np.empty((batch_size, self.length, self.hidden_size), dtype='f')
        if self.h_t is None :
            self.h_t = np.zeros((batch_size, self.hidden_size), dtype='f')
        if self.c_t is None :
            self.c_t = np.zeros((batch_size, self.hidden_size), dtype='f')

        if self.var_dropout_p > 0 and is_train :
            mask = np.random.uniform(size=self.h_t.shape) > self.var_dropout_p
            self.mask0 = mask.astype('f') * (1. / (1. - self.var_dropout_p))
            mask = np.random.uniform(size=self.h_t.shape) > self.var_dropout_p
            self.mask1 = mask.astype('f') * (1. / (1. - self.var_dropout_p))

        for t in range(self.length) :
            layer = _LSTM(*self.params)
            if self.mask0 is not None and is_train :
                self.h_t *= self.mask0
                self.c_t *= self.mask1

            self.h_t, self.c_t = layer.forward(xs[:, t, :], self.h_t, self.c_t)
            hs[:, t, :] = self.h_t
            self.layers.append(layer)
        return hs

    def backward(self, dhs) :
        Wh, Wx, b = self.params
        batch_size = dhs.shape[0]
        dxs = np.empty((batch_size, self.length, self.input_size), dtype='f')
        dh_t, dc_t = 0, 0

        grads = [0, 0, 0]
        for t in reversed(range(self.length)) :
            layer = self.layers[t]
            if self.mask0 is not None :
                dh_t *= self.mask0
                dc_t *= self.mask1
            dx_t, dh_t, dc_t = layer.backward(dhs[:, t, :] + dh_t, dc_t)
            dxs[:, t, :] = dx_t
            for i, grad in enumerate(layer.grads) :
                grads[i] += grad

        for i, grad in enumerate(grads) :
            self.grads[i][...] = grad
        self.mask0 = None
        self.mask1 = None
        return dxs

    def reset_state(self) :
        self.h_t, self.c_t = None, None


class BiLSTM :
    def __init__(self, W_h, W_x, b, var_dropout_p=0.) :
        self.fore = LSTM(W_h, W_x, b, var_dropout_p)
        self.back = LSTM(W_h, W_x, b, var_dropout_p)
        self.params = self.fore.params + self.back.params
        self.grads = self.fore.grads + self.back.grads

    def forward(self, x, is_train=True) :
        return (self.fore.forward(x) + self.back.forward(x[:, : :-1, :])[:, : :-1, :]) / 2.

    def backward(self, dout) :
        dout /= 2.
        return self.fore.backward(dout) + self.back.backward(dout[:, : :-1, :])[:, : :-1, :]

    def reset_state(self) :
        self.fore.reset_state()
        self.back.reset_state()


class Affine :
    def __init__(self, W, b) :
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.shape = None
        self.cache = None

    def forward(self, xs, is_train=True) :
        self.shape = xs.shape
        W, b = self.params
        self.cache = xs

        xs = xs.reshape(-1, xs.shape[2])
        # |x| = (bs * l, emb)
        out = np.dot(xs, W) + b
        out = out.reshape(self.shape[0], self.shape[1], -1)
        # |out| = (bs, l, vs)
        return out

    def backward(self, dout) :
        W, b = self.params
        xs = self.cache
        xs = xs.reshape(-1, self.shape[2])

        dout = dout.reshape(-1, dout.shape[2])
        db = dout.sum(axis=0)

        dxs = np.dot(dout, W.T)
        dW = np.dot(xs.T, dout)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        dxs = dxs.reshape(*self.shape)
        return dxs

class Softmax:
  def __init__(self, vocab_size):
    self.params, self.grads = [], []

    self.vocab_size = vocab_size
    self.shape = None
    self.cache = None
    self.dim = None

  def forward(self, scores, ys, is_train=True):
    # |scores| = (bs, t, vs)
    # |ys| = (bs, t)
    self.shape = scores.shape
    self.dim = scores.ndim
    if self.dim == 3:
      scores = scores.reshape(-1, self.shape[2])
      ys = ys.flatten()
    zs = softmax(scores)
    # |zs| = (bs * t, vs)
    # |ys| = (bs * y)
    loss = -np.sum(np.log(zs[np.arange(len(ys)), ys])) / (len(ys))

    self.cache = zs, ys
    return loss

  def backward(self, dout=1.):
    zs, ys = self.cache
    zs[np.arange(len(zs)), ys] -= 1
    C = (self.shape[0] * self.shape[1]) if self.dim == 3 else self.shape[0]
    zs = (zs * dout) / C
    if self.dim == 3:
      dscores = zs.reshape(self.shape)
    else:
      dscores = zs
    return dscores

class Dropout:
  def __init__(self, dropout_p, recur=False):
    self.params, self.grads = [], []
    self.dropout_p = dropout_p
    self.mask = None
    self.recur = recur

  def forward(self, x, is_train=True):
    # |x| = (bs, t, hs)
    # masking되면 0, masking 안된 노드는 신호를 조금 더 강하게
    if is_train and self.dropout_p > 0:
      if self.recur:
        mask = np.random.uniform(size=(x.shape[0], x.shape[-1])) > self.dropout_p
        self.mask = np.expand_dims(mask.astype('f'), axis=1) * (1. / (1. - self.dropout_p))
      else:
        mask = np.random.uniform(size=x.shape) > self.dropout_p
        self.mask = mask.astype('f') * (1. / (1. - self.dropout_p))
      x *= self.mask
    return x

  def backward(self, dout):
    dout *= self.mask
    self.mask = None
    return dout


class EmbeddingDropout:
  def __init__(self, dropout_p):
    self.params, self.grads = [], []
    self.dropout_p = dropout_p

  def forward(self, x, is_train=True):
    # |x| = (bs, t)
    if is_train and self.dropout_p > 0:
      choice = np.random.choice(np.arange(10000), size=int(10000 * self.dropout_p), replace=False)
      mask = np.isin(x, choice)
      x[mask] = 10000
    return x

  def backward(self, dout):
    return dout
