#encoding=utf-8

__date__ = "2017-08-03"
__author__ = "turingli"

"""
make char dataset
generate char vocab
"""

import os
import click
import codecs
import sys
import utils

@click.group()
def cli():
  pass

@click.command("convert")
@click.argument("source_dir_or_path")
@click.argument("write_dir_or_path")
def convert_dir_or_path(source_dir_or_path, write_dir_or_path):
  assert source_dir_or_path != write_dir_or_path, (source_dir_or_path, write_dir_or_path)
  if os.path.isdir(source_dir_or_path):
    if os.path.exists(write_dir_or_path) == False:
      os.makedirs(write_dir_or_path)
    path_pairs = utils.get_recur_file_paths(source_dir_or_path, write_dir_or_path, dir_pattern_list=["train", "dev", "test"])
    for source_path, target_path in path_pairs:
      convert_token_file_to_char(source_path, target_path)
  else:
    convert_token_file_to_char(source_dir_or_path, write_dir_or_path)

def convert_token_file_to_char(token_file, save_path, delimetor=" "):
  print("converting {} to char format {}".format(token_file, save_path))
  fin = codecs.open(token_file, "r", "utf-8")
  fout = codecs.open(save_path, "w", "utf-8")
  for line in fin:
    t = line.strip().split(" ")
    char_list = []
    for token in t:
      char_list.extend(list(token))
    s = delimetor.join(char_list)
    fout.write(s + "\n")
  fin.close()
  fout.close()

@click.command("char_vocab")
@click.argument("source_path_or_dir")
@click.argument("char_vocab_path")
def get_char_vocab(source_path_or_dir, char_vocab_path):
  all_paths = utils.get_dir_or_file_path(source_path_or_dir)
  char_vocab = {}
  for path in all_paths:
    f = codecs.open(path, "r", "utf-8")
    for line in f:
      chars = list(line.strip())
      for char in chars:
        if char == " ":
          continue
        if char not in char_vocab:
          char_vocab[char] = 0
        char_vocab[char] += 1
    f.close()
  print("getting {} chars".format(len(char_vocab)))
  char_vocab = sorted(char_vocab.items(), key = lambda x: x[1], reverse=True)
  print("{}-{}  {}-{}".format(char_vocab[0][0], char_vocab[0][1], char_vocab[-1][0], char_vocab[-1][1]))
  fout = codecs.open(char_vocab_path, "w", "utf-8")
  for char, count in char_vocab:
    s = " ".join([char,str(count)])
    fout.write(s + "\n")
  fout.close()

cli.add_command(convert_dir_or_path)
cli.add_command(get_char_vocab)

if __name__ == "__main__":
  cli()




