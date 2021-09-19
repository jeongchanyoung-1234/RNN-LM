import cupy as np


def softmax(x) :
    x = x - x.max(axis=1, keepdims=True)
    x = np.exp(x)
    x /= x.sum(axis=1, keepdims=True)

    return x


def sigmoid(x) :
    pos_indice = x >= 0
    neg_indice = x < 0

    new_x = np.zeros_like(x, float)
    new_x[pos_indice] = 1 / (1 + np.exp(-x[pos_indice]))
    new_x[neg_indice] = np.exp(x[neg_indice]) / (1 + np.exp(x[neg_indice]))

    return new_x


def cross_entropy(y, y_hat) :
    return -np.sum(np.log(y_hat[np.arange(len(y)), y] + 1e-4)) / len(y)


def cross_entropy(t, y) :
    if y.ndim == 1 :
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 정답 데이터가 원핫 벡터일 경우 정답 레이블 인덱스로 변환
    if t.size == y.size :
        t = t.argmax(axis=1)

    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def clip_grads(grads, max_norm) :
    total_norm = 0
    for grad in grads :
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm)
    if rate < 1 :
        for grad in grads :
            grad *= rate


def get_norm(params, grads, norm_type=2.) :
    p_norm = 0
    for param in params :
        p_norm += (param ** norm_type).sum()
    p_norm **= (1. / norm_type)

    g_norm = 0
    for grad in grads :
        g_norm += (grad ** norm_type).sum()
    g_norm **= (1. / norm_type)

    if np.isnan(p_norm) or np.isinf(p_norm) :
        p_norm = 0.

    if np.isnan(g_norm) or np.isinf(g_norm) :
        g_norm = 0.

    return p_norm, g_norm