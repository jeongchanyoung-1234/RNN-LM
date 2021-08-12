import argparse

import ptb
from models.lstmlm import LSTMLM
from module.optimizer import *
from module.trainer import Trainer
from module.functions import get_ppl


def define_argparser():
  p = argparse.ArgumentParser()

  p.add_argument('--hidden_size', type=int, default=650)
  p.add_argument('--word_vec_size', type=int, default=650)
  p.add_argument('--batch_size', type=int, default=256)
  p.add_argument('--trunc_size', type=int, default=35)
  p.add_argument('--valid_batch_size', type=int, default=128)
  p.add_argument('--lr', type=float, default=20)
  p.add_argument('--epochs', type=int, default=40)
  p.add_argument('--max_grad_norm', type=float, default=30)
  p.add_argument('--dropout', type=float, default=.5)
  p.add_argument('--var_dropout', type=float, default=0.05)
  p.add_argument('--weight_decay', type=float, default=0.25)
  p.add_argument('--warmup_epoch', type=int, default=0)
  p.add_argument('--verbose', type=int, default=1)
  p.add_argument('--early_stopping', type=int, default=5)
  p.add_argument('--train_size', type=int, default=0)
  p.add_argument('--validation', action='store_true')
  p.add_argument('--bidirectional', action='store_true')
  p.add_argument('--load_fn', type=str, default='lstmlm.pkl')

  config = p.parse_args()
  return config

def main(config):
    corpus, word_to_id, id_to_word = ptb.load_data('train')
    corpus_val, _, _ = ptb.load_data('val')
    corpus_test, _, _ = ptb.load_data('test')
    valid_data = None

    if config.train_size > 0:
        corpus = corpus[:config.train_size]
        corpus_val = corpus[:int(config.train_size * .2)]

    xs, ys = corpus[:-1], corpus[1:]
    if config.validation: valid_data = corpus_val

    vocab_size = len(word_to_id)
    model = LSTMLM(vocab_size, 
                   config.word_vec_size, 
                   config.hidden_size,
                   config.dropout,
                   config.var_dropout,
                   config.bidirectional)
    
    if config.load_fn is not None:
      model.load_params(config.load_fn)

    optimizer = SGD(lr=config.lr)

    trainer = Trainer(config, model, optimizer)
    trainer.train(xs, ys, valid_data=valid_data)

    trainer.plot_result(config.validation)
    trainer.save_model('lstmlm.pkl')

    print('TEST PPL', float(get_ppl(trainer.model, corpus_test, config.valid_batch_size, config.trunc_size)))

config = define_argparser()
main(config)