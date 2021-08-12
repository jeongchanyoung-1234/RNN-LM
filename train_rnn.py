import argparse

import matplotlib.pyplot as plt

import ptb
from models.rnnlm import RNNLM
from module.optimizer import *
from module.trainer import Trainer


def define_argparser():
  p = argparse.ArgumentParser()

  p.add_argument('--hidden_size', type=int, default=100)
  p.add_argument('--word_vec_size', type=int, default=100)
  p.add_argument('--batch_size', type=int, default=100)
  p.add_argument('--trunc_size', type=int, default=5)
  p.add_argument('--valid_batch_size', type=int, default=16)
  p.add_argument('--lr', type=float, default=1e-2)
  p.add_argument('--epochs', type=int, default=100)
  p.add_argument('--max_grad_norm', type=float, default=30)

  p.add_argument('--weight_decay', type=float, default=0)
  p.add_argument('--warmup_epochs', type=int, default=0)
  p.add_argument('--verbose', type=int, default=20)
  p.add_argument('--early_stopping', type=int, default=0)
  p.add_argument('--train_size', type=int, default=1000)

  config = p.parse_args()
  return config

def main(config) :
    corpus, word_to_id, id_to_word = ptb.load_data('train')
    corpus_val, _, _ = ptb.load_data('valid')
    if config.train_size > 0 :
        corpus = corpus[:config.train_size]
        corpus_val = corpus_val[:int(config.train_size * 0.2)]
    if not config.validation : corpus_val = None

    xs, ys = corpus[:-1], corpus[1 :]
    vocab_size = len(word_to_id)

    model = RNNLM(vocab_size, config.word_vec_size, config.hidden_size, config.trunc_size)

    optimizer = Adam(lr=config.lr)
    trainer = Trainer(config, model, optimizer)
    trainer.train(xs, ys, valid_data=corpus_val)

    plt.plot(trainer.ppl_list, label='train')
    if config.validation : plt.plot(trainer.valid_ppl_list, label='valid')
    plt.title('Result')
    plt.xlabel('Epochs')
    plt.ylabel('PPL')
    plt.ylim(0, 350)
    plt.show()

if __name__ == '__main__':
    config = define_argparser()
    main(config)