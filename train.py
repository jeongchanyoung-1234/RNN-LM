import argparse

import matplotlib.pyplot as plt

import ptb
from models.rnnlm import RNNLM
from models.lstmlm import LSTMLM
from module.optimizer import *
from module.trainer import Trainer


def define_argparser():
  p = argparse.ArgumentParser()

  p.add_argument('--model', type=str, default='lstm')
  p.add_argument('--hidden_size', type=int, default=100)
  p.add_argument('--word_vec_size', type=int, default=100)
  p.add_argument('--word_vec_size', type=int, default=100)
  p.add_argument('--batch_size', type=int, default=100)
  p.add_argument('--trunc_size', type=int, default=5)
  p.add_argument('--valid_batch_size', type=int, default=16)
  p.add_argument('--dropout', type=float, default=.35)
  p.add_argument('--var_dropout', type=float, default=.2)
  p.add_argument('--emb_dropout', type=float, default=.1)
  p.add_argument('--warmup_epoch', type=int, default=0)
  p.add_argument('--lr', type=float, default=1e-2)
  p.add_argument('--lr_decay', type=float, default=0)
  p.add_argument('--lr_decay_factor', type=float, default=0)
  p.add_argument('--lr_decay_epoch', type=int, default=0)
  p.add_argument('--epochs', type=int, default=100)
  p.add_argument('--max_grad_norm', type=float, default=30)

  p.add_argument('--weight_decay', type=float, default=0)
  p.add_argument('--warmup_epochs', type=int, default=0)
  p.add_argument('--verbose', type=int, default=20)
  p.add_argument('--early_stopping', type=int, default=0)
  p.add_argument('--train_size', type=int, default=1000)
  p.add_argument('--load_fn', type=str, default=None)
  p.add_argument('--save_fn', type=str, default=None)
  p.add_argument('--validation', action='store_true')
  p.add_argument('--bidirectional', action='stroe_true')

  config = p.parse_args()
  return config


def main(config) :
    opts = {'sgd': SGD, 'momentum': Momentum, 'adam': Adam}
    corpus, word_to_id, id_to_word = ptb.load_data('train')
    corpus_val, _, _ = ptb.load_data('valid')
    corpus_test, _, _ = ptb.load_data('test')
    valid_data = None

    if config.train_size > 0 :
        corpus = corpus[:config.train_size]
        corpus_val = corpus_val[:int(config.train_size * .2)]
    if not config.validation : corpus_val = None

    if config.emb_dropout > 0 :
        word_to_id['<mask>'] = 10000
        id_to_word[10000] = '<mask>'

    xs, ys = corpus[:-1], corpus[1 :]
    if config.validation : valid_data = corpus_val
    vocab_size = len(word_to_id)

    if config.model == 'rnn':
        model = RNNLM(vocab_size,
                      config.word_vec_size,
                      config.hidden_size,
                      config.trunc_size,
                      rnn=config.model)
    elif config.model == 'lstm':
        model = LSTMLM(vocab_size,
                       config.word_vec_size,
                       config.hidden_size,
                       config.dropout,
                       config.var_dropout,
                       config.emb_dropout,
                       config.bidirectional)

    if config.load_fn is not None :
        model.load_params(config.load_fn)

    optimizer = opts[config.optimizer.lower()](lr=config.lr)
    trainer = Trainer(config, model, optimizer)
    trainer.train(xs, ys, valid_data=valid_data)

    plt.plot(trainer.ppl_list, label='train')
    if config.validation : plt.plot(trainer.valid_ppl_list, label='valid')
    plt.title('Result')
    plt.xlabel('Epochs')
    plt.ylabel('PPL')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    config = define_argparser()
    main(config)
