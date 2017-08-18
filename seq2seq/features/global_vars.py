#encoding=utf-8

import collections
import tensorflow as tf

class SpecialWordsClass(object):

  def __init__(self, special_words=["PAD","UNK", "SEQUENCE_START", "SEQUENCE_END", "PARA_START", "PARA_END"] ):

    for word in special_words:
      setattr(self, word, word)
    self._total_words = special_words

SpecialWordsIns = SpecialWordsClass()
SpecialWords = SpecialWordsIns._total_words
SpecialVocab = collections.namedtuple("SpecialVocab", SpecialWordsIns._total_words)

source_keys_to_features = {
  "source_tokens": tf.VarLenFeature(tf.string),
  "source_ids": tf.VarLenFeature(tf.int64),
  "extend_source_ids": tf.VarLenFeature(tf.int64),
  "source_oov_list": tf.VarLenFeature(tf.string),
  "source_oov_nums": tf.FixedLenFeature([], tf.int64),
  "source_ners": tf.VarLenFeature(tf.string),
  "source_ner_ids": tf.VarLenFeature(tf.int64),
  "source_postags": tf.VarLenFeature(tf.string),
  "source_pos_ids": tf.VarLenFeature(tf.int64),
  "source_tfidfs": tf.VarLenFeature(tf.float32),
}

aliment_keys_to_features = {
  "aliment": tf.VarLenFeature(tf.int64)
}

target_keys_to_features = {
  "target_tokens": tf.VarLenFeature(tf.string),
  "target_ids": tf.VarLenFeature(tf.int64),
  "extend_target_ids": tf.VarLenFeature(tf.int64),
  "target_ner_ids": tf.VarLenFeature(tf.int64),
  "target_ners": tf.VarLenFeature(tf.string),
}

source_feature_keys = list(source_keys_to_features.keys())
target_feature_keys = list(target_keys_to_features.keys())

int64_keys = ["source_ids", "extend_source_ids", "source_oov_nums", "source_ner_ids", "source_pos_ids", "target_ids",
              "extend_target_ids", "target_ner_ids", "aliment"]
float_keys = ["source_tfidfs"]
bytes_keys = ["source_ners", "source_postags", "source_tokens", "source_oov_list", "target_tokens", "target_ners"]

if __name__ == "__main__":
  print(SpecialVocab._fields)