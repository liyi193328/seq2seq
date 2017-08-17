#encoding=utf-8


import os
import codecs
import sys
import six
import pyltp

import math

from pyltp import Segmentor
from pyltp import SentenceSplitter
from pyltp import Postagger
from pyltp import NamedEntityRecognizer

import seq2seq.features.utils as utils
from seq2seq.features import SpecialWords

HOME_PATH = os.path.expanduser("~")
default_ltp_data_path = os.path.join(HOME_PATH, "software/LTP/ltp_data")
LTP_DATA_DIR = os.environ.get("LTP_DATA_DIR", default_ltp_data_path)
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`

segmentor = Segmentor()  # 初始化实例
segmentor.load(cws_model_path)  # 加载模型
postagger = Postagger() # 初始化实例
postagger.load(pos_model_path)  # 加载模型
recognizer = NamedEntityRecognizer() # 初始化实例
recognizer.load(ner_model_path)  # 加载模型

def SentenceSplit(para):
  sents = SentenceSplitter.split(para)
  return sents

def cut_sentence(words):
  start = 0
  i = 0  # 记录每个字符的位置
  sents = []
  punt_list = u'.!?;~。！？～'  # string 必须要解码为 unicode 才能进行匹配
  for word in words:
    if six.PY2 and type(word) == str:
      word = word.decode("utf-8")
    # print(type(word))
    if word in punt_list:
      sents.append(words[start:i + 1])
      start = i + 1  # start标记到下一句的开头
      i += 1
    else:
      i += 1  # 若不是标点符号，则字符位置继续前移
  if start < len(words):
    sents.append(words[start:])  # 这是为了处理文本末尾没有标点符号的情况
  return sents

def Postags(words):
  if type(words) != list:
    words = [words]
  if six.PY2:
    if type(words[0]) == unicode:
      words = [v.encode("utf-8") for v in words]
  postags = list(postagger.postag(words))  # 词性标注
  return postags

def NamedEntityRecogize(words, postags=None):
  if type(words) != list:
    words = [words]
  if six.PY2:
    words = [v.encode("utf-8") for v in words]
  if postags is None:
    postags = Postags(words)

  netags = list(recognizer.recognize(words, postags))
  return netags

def get_all_ner_tag(file_path, save_path):
  f = codecs.open(file_path, "r", "utf-8")
  ner_tags = set()
  for line in f:
    line = line.strip()
    words = line.split()
    tags = NamedEntityRecogize(words)
    ner_tags = ner_tags.union(set(tags))
  f.close()
  fout = codecs.open(save_path, "w","utf-8")
  for ner in ner_tags:
    fout.write(ner+"\n")
  fout.close()

class Tfidf(object):

  def __init__(self, path, word_index=0, value_index=2, special_words=SpecialWords, default=0.0):

    self._path = path
    f = codecs.open(path, "r", "utf-8")
    self._word2value = {}

    for word in special_words:
      self._word2value[word] = default

    for line in f:
      cells = line.strip().split("\t")
      try:
        word = cells[word_index]
        value = cells[value_index]
        self._word2value[word] = float(value)
      except IndexError:
        continue

  def get(self, word, default=0.0):
    if word not in self._word2value:
      return default
    return self._word2value[word]

  def encode(self, words, default=0.0):
    ans = [self.get(word, default=0.0) for word in words]
    return ans


class Tfidf_online(object):

  def __init__(self, path_or_dir):
    self._paths = utils.get_dir_or_file_path(path_or_dir)
    doc_word_cnts = []
    token_list = []
    for path in self._paths:
      f = codecs.open(path, "r", "utf-8")
      for line in f:
        tokens = line.strip().split()
        word_cnt = {}
        for token in tokens:
          token = token.strip()
          if len(token) == 0:
            continue
          if token not in word_cnt:
            word_cnt[token] = 0
          word_cnt[token] += 1
          doc_word_cnts.append(word_cnt)
    self._doc_word_cnts = doc_word_cnts

  def word_cnt_map(self, token_list):
    word_cnt = {}
    for token in token_list:
      token = token.strip()
      if len(token) == 0:
        continue
      if token not in word_cnt:
        word_cnt[token] = 0
      word_cnt[token] += 1
    return word_cnt

  def tf(self, word, word_dict_cnt):
    if word not in word_dict_cnt:
      return 0.0
    return word_dict_cnt[word] / len(word_dict_cnt)

  def n_containing(self, word, doc_word_cnts):
    return sum(1 for doc in doc_word_cnts if word in doc)

  def idf(self, word, doc_word_cnts):
    return math.log(len(doc_word_cnts) / (1 + self.n_containing(word, doc_word_cnts)))

  def tfidf(self, word, word_cnt, doc_word_cnts):
    return self.tf(word, word_cnt) * self.idf(word, doc_word_cnts)

  def get_words_tfidf(self, word_list):
    word_cnt = self.word_cnt_map(word_list)
    value_list = []
    for word in word_list:
      v = self.tfidf(word, word_cnt, self._doc_word_cnts)
      value_list.append(v)
    return value_list

if __name__ == "__main__":
  words = ['朴槿惠','被' ,'调查','了', "END"]
  # print(NamedEntityRecogize(words))
  # get_all_ner_tag("/home/bigdata/active_project/run_tasks/query_rewrite/debug/data/train/sources.txt", "/home/bigdata/nlp_data/ner_tags.txt")
  print(Postags(words))
  print(NamedEntityRecogize(words))
