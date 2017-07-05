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


from seq2seq.graph_utils import templatemethod

@templatemethod("trial")
def trial(x):
    def f(x):
      w = tf.get_variable('w', [])
      tf.reduce_sum(x) * w
    return f

def my_trail(x, share_variable_name):
  var1 = tf.get_variable(share_variable_name, shape=[])
  return tf.reduce_sum(x) * var1

template_my = tf.make_template("template_my", my_trail, share_variable_name="my_v")

def test_trial():
  y = tf.placeholder(tf.float32, [None])
  z = tf.placeholder(tf.float32, [None])
  with tf.variable_scope("my"):
    f = trial(y)
    a_z = trial(z)

  # a_y = template_my(y)
  # a_z = template_my(z)

  s = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  print(tf.global_variables())
  # print(v_0.eval())
  print(a_y.eval(feed_dict={y: [1.1, 1.9]}))
  # print(v_1.eval())
  print(a_z.eval(feed_dict={z: [1.9, 1.1]}))

if __name__ == "__main__":

  test_trial()


