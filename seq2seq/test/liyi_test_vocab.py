#encoding=utf-8


import numpy as np
import tensorflow as tf

from tensorflow import gfile

from seq2seq.data.vocab import create_vocabulary_lookup_table


# x = create_vocabulary_lookup_table(r"E:\active_project\run_tasks\seq2seq\question_gen\train\shared.vocab.txt")

def t(filename, default_value=None):
  """Creates a lookup table for a vocabulary file.

  Args:
    filename: Path to a vocabulary file containg one word per line.
      Each word is mapped to its line number.
    default_value: UNK tokens will be mapped to this id.
      If None, UNK tokens will be mapped to [vocab_size]

    Returns:
      A tuple (vocab_to_id_table, id_to_vocab_table,
      word_to_count_table, vocab_size). The vocab size does not include
      the UNK token.
    """
  if not gfile.Exists(filename):
    raise ValueError("File does not exist: {}".format(filename))

  # Load vocabulary into memory
  with gfile.GFile(filename) as file:
    vocab = list(line.strip("\n") for line in file)
  vocab_size = len(vocab)

  has_counts = len(vocab[0].split("\t")) == 2
  if has_counts:
    y = []
    for _ in vocab:
        x = _.split("\t")
        if len(x) != 2:
            print("error:")
            print(_)
            break
        y.append(x)
    # vocab, counts = zip(*[_.split("\t") for _ in vocab])
    vocab, counts = zip(*y)
    counts = [float(_) for _ in counts]
    vocab = list(vocab)
  return vocab

class f(object):

  def __init__(self,x,y):
    self._x = x
    self._y = y

  def _build(self, c,d):
    print("in build")
    print(c, d)




if __name__ == "__main__":

  a = f(1,2)
  a._build(3,4)



