#encoding=utf-8


class SpecialWordsClass(object):

  def __init__(self):

    self.UNK = "UNK"
    self.PAD = "PAD"
    self.SEQUENCE_START = "SEQUENCE_START"
    self.SEQUENCE_END = "SEQUENCE_END"
    self.PARA_START = "PARA_START"
    self.PARA_END = "PARA_END"
    self._total_words = ["PAD","UNK", "SEQUENCE_START", "SEQUENCE_END", "PARA_START", "PARA_END"]

SpecialWords = SpecialWordsClass()

