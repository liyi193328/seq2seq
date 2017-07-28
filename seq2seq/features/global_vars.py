#encoding=utf-8

import collections
class SpecialWordsClass(object):

  def __init__(self, special_words=["PAD","UNK", "SEQUENCE_START", "SEQUENCE_END", "PARA_START", "PARA_END"] ):

    for word in special_words:
      setattr(self, word, word)
    self._total_words = special_words

SpecialWordsIns = SpecialWordsClass()
SpecialVocab = collections.namedtuple("SpecialVocab", SpecialWordsIns._total_words)

if __name__ == "__main__":
  print(SpecialVocab._fields)