import time
import pickle
from itertools import combinations

import cupy as np
import matplotlib.pyplot as plt
from functions import clip_grads, get_norm


class Trainer :
    def __init__(self, config, model, optimizer) :
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.best_ppl = np.inf
        self.best_epoch = None

        self.ppl_list = []
        self.valid_ppl_list = []

        self.time_idx = None
        self.valid_time_idx = None

    def validation(self, corpus, batch_size, time_size) :
        corpus_size = len(corpus)
        total_loss, loss_cnt = 0, 0
        max_iters = (corpus_size - 1) // (batch_size * time_size)
        jump = (corpus_size - 1) // batch_size

        for iters in range(max_iters) :
            xs = np.zeros((batch_size, time_size), dtype=np.int32)
            ts = np.zeros((batch_size, time_size), dtype=np.int32)
            time_offset = iters * time_size
            offsets = [time_offset + (i * jump) for i in range(batch_size)]
            for t in range(time_size) :
                for i, offset in enumerate(offsets) :
                    xs[i, t] = corpus[(offset + t) % corpus_size]
                    ts[i, t] = corpus[(offset + t + 1) % corpus_size]

            loss = self.model.forward(xs, ts, is_train=False)
            total_loss += loss
        ppl = np.exp(total_loss / max_iters)
        return ppl

    def weight_tying(self, params, grads) :
        params, grads = params[:], grads[:]

        while True :
            length = len(params)
            for a, b in combinations(np.arange(length - 1), 2) :
                a, b = int(a), int(b)
                if params[a].shape == params[b].shape :
                    if params[a] is params[b] :
                        grads[a] += grads[b]
                        params.pop(b)
                        grads.pop(b)
                        break
                elif params[a].shape == params[b].T.shape :
                    if np.all(params[a] == params[b].T) :
                        grads[a] += grads[b].T
                        params.pop(b)
                        grads.pop(b)
                        break
            else :
                break

        return params, grads

    def get_batch(self, x, t, batch_size, time_size, type='train') :
        batch_x = np.empty((batch_size, time_size), dtype='i')
        batch_t = np.empty((batch_size, time_size), dtype='i')

        data_size = len(x)
        jump = data_size // batch_size
        offsets = [i * jump for i in range(batch_size)]

        time_idx = self.time_idx if type == 'train' else self.valid_time_idx
        for time in range(time_size) :
            for i, offset in enumerate(offsets) :
                batch_x[i, time] = x[(offset + time_idx) % data_size]
                batch_t[i, time] = t[(offset + time_idx) % data_size]
            time_idx += 1
        if type == 'train' :
            self.time_idx = time_idx
        elif type == 'valid' :
            self.valid_time_idx = time_idx
        return batch_x, batch_t

    def reset_state(self) :
        for layer in self.model.rnn_layers :
            layer.reset_state()

    def plot_result(self, validation) :
        plt.plot(self.ppl_list, label='train ppl')
        if validation :
            plt.plot(self.valid_ppl_list, label='valid ppl')
        plt.legend()
        plt.title('Training result')
        plt.xlabel('Epochs')
        plt.ylabel('PPL')
        plt.show()

    def save_model(self, save_fn) :
        params = {}
        params['params'] = self.model.params
        with open(save_fn, 'wb') as f :
            pickle.dump(params, f)

    def train(self, xs, ys, valid_data=None) :
        batch_size = self.config.batch_size
        trunc_size = self.config.trunc_size
        model, optimizer = self.model, self.optimizer

        total_loss = 0.
        loss_count = 0.

        iters = (len(xs) // (batch_size * trunc_size))

        start = time.time()
        if self.config.early_stopping > 0 :
            patience = self.config.early_stopping
            print(
                f'|Message| Training will be automatically stopped if no improvement during {self.config.early_stopping} epochs')
        for epoch in range(1, self.config.epochs + 1) :
            if self.config.lr_decay_factor > 0 and self.config.lr_decay_epoch <= epoch :
                optimizer.lr /= self.config.lr_decay_factor
            self.time_idx = 0
            self.valid_time_idx = 0

            # Warmup stage
            if self.config.warmup_epoch > 0 :
                if epoch <= self.config.warmup_epoch :
                    warmup_lr = self.config.lr * (epoch / self.config.warmup_epoch)
                    print('Warmup stage - current learning rate {}'.format(warmup_lr))
                    optimizer.lr = warmup_lr

            for iter in range(iters) :
                batch_X, batch_y = self.get_batch(xs, ys, self.config.batch_size, self.config.trunc_size)
                loss = model.forward(batch_X, batch_y, is_train=True)
                model.backward(1.)
                # weight_tying
                params, grads = self.weight_tying(model.params, model.grads)
                # gradient_clipping
                clip_grads(grads, self.config.max_grad_norm)
                p_norm, g_norm = get_norm(params, grads)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

            avg_loss = total_loss / loss_count
            total_loss = 0.
            loss_count = 0.
            avg_ppl = np.exp(avg_loss)
            self.ppl_list.append(avg_ppl)

            if valid_data is not None :
                self.reset_state()
                avg_valid_ppl = self.validation(valid_data, self.config.valid_batch_size, self.config.trunc_size)
                self.valid_ppl_list.append(avg_valid_ppl)
                self.reset_state()

            ppl = avg_ppl if valid_data is None else avg_valid_ppl
            if ppl < self.best_ppl :
                self.best_ppl = ppl
                self.best_epoch = epoch
                patience = self.config.early_stopping
                if self.config.save_fn is not None :
                    self.save_model(self.config.save_fn)
            else :
                if self.config.lr_decay > 0 :
                    if self.config.warmup_epoch <= 0 or self.config.warmup_epoch > 0 and epoch > self.config.warmup_epoch :
                        optimizer.lr *= self.config.lr_decay
                        print(f'Learning rate decayed - Current {optimizer.lr}')
                if self.config.early_stopping > 0 :
                    patience -= 1
                    if patience == 0 :
                        print(f'Training stopped early at {epoch} epochs')
                        break

            end = time.time()
            if epoch % self.config.verbose == 0 :
                if valid_data is None :
                    print(
                        '''|EPOCH ({}/{}) train_loss={:.4f} train_ppl={:4.2f} best_ppl={:4.2f} |param|={:.4f} |grad|={:.4f} ({:.4f}sec)'''.format(
                            epoch, self.config.epochs,
                            avg_loss, avg_ppl, self.best_ppl,
                            p_norm, g_norm, end - start
                        ))
                else :
                    print(
                        '''|EPOCH ({}/{}) train_loss={:.4f} train_ppl={:4.2f} valid_ppl={:4.2f} best_ppl={:4.2f} |param|={:.4f} |grad|={:.4f} ({:.4f}sec)'''.format(
                            epoch, self.config.epochs,
                            avg_loss, avg_ppl,
                            avg_valid_ppl, self.best_ppl,
                            p_norm, g_norm,
                            end - start
                        ))

        print()
        print('=' * 10, 'RESULT', '=' * 10)
        print('BEST PPL', self.best_ppl)
        print('BEST EPOCH', self.best_epoch)