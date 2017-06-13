#encoding=utf-8


import os
import sys
import codecs
import six
import shutil
import subprocess
import numpy as np
from os.path import join


import click

global_gen_vocb_script_path = None

try:
  import seq2seq
  project_path = os.path.dirname(seq2seq.__path__[0])
  global_gen_vocb_script_path = join(project_path, "bin/tools/generate_vocab.py")
except ImportError:
  print("May add seq2seq dir into pythonpath")

def filter_pos(file_path, save_path):
    f = codecs.open(file_path, "r", "utf-8")
    lines = f.readlines(f)
    f.close()
    f = codecs.open(save_path, "w", "utf-8")
    new_lines = []
    for line in lines:
        t = line.strip().split("\t")
        try:
            if int(t[0]) == 1:
                x = "\t".join([t[1], t[2]]) + "\n"
                f.write(x)
        except Exception:
            print("error:", line)
    f.close()

def filter_illegal(path, in_place=True, newpath=None):
    if in_place is False:
        assert newpath is not None
    print("read path {}".format(path))
    f = codecs.open(path, "r", "utf-8", errors="ignore")
    lines = f.readlines()
    newlines = []
    vocab_dict = {}
    for i, line in enumerate(lines):
        t = line.strip().split("\t")
        if len(t) != 2:
            continue
        if t[0] in vocab_dict:
            print(line)
            print("{}th line: dup {}".format(i, t[0]))
            continue
        vocab_dict[t[0]] = t[1]
        newlines.append(line)
    f.close()
    if newpath is None:
        newpath = path
    print("total lens: {}".format(len(lines)))
    write_list_to_file(newlines, newpath)

def read_file(file_path, in_memory=True, verb=True):
    f = codecs.open(file_path, "r","utf-8")
    if in_memory:
        return f.readlines()
    else:
        raise NotImplemented("read one line every time")

def write_list_to_file(d, file_path, verb=True):
    assert type(d) == list
    if os.path.exists(os.path.dirname(file_path)) == False:
        os.makedirs(os.path.dirname(file_path))
    f = codecs.open(file_path, "w","utf-8")
    for x in d:
        f.write(x.strip() + "\n")
    f.close()
    if verb:
        print("write to {}".format(file_path))

def merge_files(file_list, outpath):
    q = []
    for i, file in enumerate(file_list):
        q.extend(read_file(file))
    write_list_to_file(q, outpath)

def sample_ques_pairs(file_path, sample_size=None):
    lines = read_file(file_path)
    q0s, q1s = [], []
    for line in lines:
        t = line.strip().split("\t")
        if len(t) < 2:
            continue
        q0s.append(t[0].strip())
        q1s.append(t[1].strip())
    if sample_size is None:
        return q0s, q1s
    s = len(q0s)
    sample_index = np.random.choice(list(range(s)), sample_size)
    sample_q0 = [q0s[i] for i in sample_index]
    sample_q1 = [q1s[i] for i in sample_index]
    return sample_q0, sample_q1

def sample_pairs(post_file_path, res_file_path, sample_size=None):
    
    posts, responses = read_file(post_file_path), read_file(res_file_path)
    assert len(posts) == len(responses)
    s = len(posts)
    if sample_size is None:
        sample_index = list(range(len(posts)))
        sample_posts = posts
        sample_responses = responses
    else:
        sample_index = np.random.choice(list(range(s)), sample_size)
        sample_posts = [posts[i] for i in sample_index]
        sample_responses = [responses[i] for i in sample_index]
    return sample_posts, sample_responses

def write_some_data(source, target, out_dir):
    source_path = join(out_dir, "sources.txt")
    target_path = join(out_dir, "targets.txt")
    write_list_to_file(source, source_path)
    write_list_to_file(target, target_path)

def make_train_test_dev(source, target, out_dir,ratio_split=None):

    if ratio_split is None:
        train_ratio = 0.95
        dev_ratio = 1.0
        test_ratio = 1.0
    else:
        train_ratio, dev_ratio, test_ratio = ratio_split

    train_dir = join(out_dir, "train")
    dev_dir = join(out_dir, "dev")
    test_dir = join(out_dir, "test")

    train_index = int( len(source) * train_ratio )
    dev_index = int(len(source) * dev_ratio)

    write_some_data( source[0:train_index], target[0:train_index], train_dir)
    write_some_data( source[train_index:dev_index], target[train_index:dev_index] , dev_dir)
    write_some_data( source[dev_index:], target[dev_index:] , test_dir)

def generate_vocb(script_path, tokenized_file, vocb_path):
    script_path = os.path.abspath(script_path)
    tokenized_file = os.path.abspath(tokenized_file)
    vocb_path = os.path.abspath(vocb_path)
    if os.path.exists(os.path.dirname(vocb_path)) == False:
        os.makedirs(os.path.dirname(vocb_path))
    cmd = "python {} < {} > {} --min_frequency 4".format(script_path, tokenized_file, vocb_path)
    print("lanch : {}...".format(cmd))
    r=subprocess.run(cmd, shell=True, check=True)
    print(r)
    filter_illegal(vocb_path)

def generate_parallel_vocbs(generate_vocb_script_path, data_dir):
    train_source_path = join(data_dir, "train/sources.txt")
    train_source_vocab_path = join(data_dir, "vocab/vocab.sources.txt")
    train_target_path = join(data_dir, "train/targets.txt")
    train_target_vocab_path = join(data_dir, "vocab/vocab.sources.txt")
    merge_path = join(data_dir, "train/merge_train_target.txt")
    shared_vocab_path = join(data_dir, "vocab/shared.vocab.txt")
    merge_files([train_source_path, train_target_path], merge_path)
    generate_vocb(generate_vocb_script_path, merge_path, shared_vocab_path)
    os.remove(merge_path)

def get_source_target(qs, keys, xs, ys,keep_one=False):
  sources = []
  targets = []
  for key in keys:
    vs = qs[key]
    if keep_one is True:
      vs = [vs[0]]
    sources.extend([xs[v] for v in vs])
    targets.extend([ys[v] for v in vs])
  return sources, targets

def get_unique_ques(file_path,save_path, sample_size=None, add_dual=True):
  from os.path import join
  if os.path.exists(os.path.dirname(save_path)) == False:
    os.makedirs(os.path.dirname(save_path))
  lines = codecs.open(file_path, "r", "utf-8").readlines()
  f = codecs.open(save_path,"w", "utf-8")
  qs_set = set()
  for i, line in enumerate(lines):
    t = line.strip().split("\t")
    x = t[0].strip()
    y = t[1].strip()
    if x not in qs_set:
      qs_set.add(x)
      f.write(x + "\n")
    if add_dual is True:
      if y not in qs_set:
        qs_set.add(y)
        f.write(y+"\n")
  f.close()

@click.group()
def cli():
    pass

@click.command()
@click.argument("source_data_path")
@click.argument("save_data_dir")
@click.option("--sample_size", default=None, help="sample size, None mean all")
@click.option("--ratios", default="0.95,1.0,1.0", help="train,dev,test split ratio")
@click.option("--add_dual/--no-add_dual", deault=False, help="whether add dual pair from target to source")
@click.option("--seq2seq_path", default=None, help="seq2seq dir")
def make_sep_datasets(source_data_path, save_data_dir, ratios="0.95,1.0,1.0",
                      keep=None, sample_size=None,
                      add_dual=False, seq2seq_path=None):

  from os.path import join
  global  global_gen_vocb_script_path

  if os.path.exists(save_data_dir) == False:
    os.makedirs(save_data_dir)

  ratio_list = [float(v) for v in ratios.split(",")]
  if len(ratio_list) != 3:
      raise ValueError("error in {}, must sure 3 elements like 0.95,1.0,1.0".format(ratios))

  lines = codecs.open(source_data_path,"r","utf-8").readlines()
  qs = {}
  xs = []
  ys = []

  for i, line in enumerate(lines):
    t = line.strip().split("\t")
    x = t[0].strip()
    y = t[1].strip()
    xs.append(x)
    ys.append(y)
    if x not in qs:
     qs[x] = []
    if keep is None:
      qs[x].append(i)
    else:
      if len(qs[x]) > keep:
        continue

  q_num = len(qs)
  ques_list = list(qs.keys())
  train_index = int(q_num * ratio_list[0])
  eval_index = int(q_num * ratio_list[1])
  test_index = int(q_num * ratio_list[2])

  train_dir = join(save_data_dir, "train")
  eval_dir = join(save_data_dir, "dev")
  test_dir = join(save_data_dir, "test")

  train_keys, eval_keys, test_keys = ques_list[0:train_index], \
                                     ques_list[train_index:eval_index],\
                                     ques_list[eval_index:test_index]

  train_s, train_t = get_source_target(qs, train_keys, xs, ys)
  eval_s, eval_t = get_source_target(qs, eval_keys, xs, ys, keep_one=True)
  test_s, test_t = get_source_target(qs, test_keys, xs, ys, keep_one=True)

  if add_dual:
    train_s.extend(train_t)
    train_t.extend(train_s)
    eval_s.extend(eval_t)
    eval_t.extend(eval_s)
    test_s.extend(test_t)
    test_t.extend(test_s)

  write_some_data(train_s, train_t, train_dir)
  write_some_data(eval_s, eval_t, eval_dir)
  write_some_data(test_s, test_t, test_dir)

  if seq2seq_path is not None:
    global_gen_vocb_script_path = join(seq2seq_path, "bin/tools/generate_vocab.py")
  elif global_gen_vocb_script_path is None:
    raise ValueError("No seq2seq found")

  generate_parallel_vocbs(global_gen_vocb_script_path, save_data_dir)
  get_unique_ques(source_data_path, join(save_data_dir, "all_ques.txt"), add_dual=True)

@click.command()
@click.argument("source_data_path")
@click.argument("save_data_dir")
@click.option("--sample_size", default=None, help="sample size, None mean all")
@click.option("--ratios", default="0.95,1.0,1.0", help="train,dev,test split ratio")
@click.option("--seq2seq_path", default=None, help="seq2seq dir")
def make_inter_datasets(source_data_path, save_data_dir,ratios="0.95,1.0,1.0", sample_size=None,seq2seq_path=None):

    from os.path import join
    global global_gen_vocb_script_path

    ratio_list = [float(v) for v in ratios.split(",")]
    if len(ratio_list) != 3:
        raise ValueError("error in {}, must sure 3 elements like 0.95,1.0,1.0".format(ratios))

    if seq2seq_path is not None:
      global_gen_vocb_script_path = join(seq2seq_path, "bin/tools/generate_vocab.py")
    elif global_gen_vocb_script_path is None:
      raise ValueError("No seq2seq found")

    sample_sources, sample_targets = sample_ques_pairs(source_data_path, sample_size)
    make_train_test_dev(sample_sources, sample_targets, save_data_dir, ratio_split=ratio_list)

    generate_parallel_vocbs(global_gen_vocb_script_path, save_data_dir)

cli.add_command(make_inter_datasets)
cli.add_command(make_sep_datasets)

if __name__ == '__main__':

    # source_path = join(data_dir, "stc_weibo_train_post")
    # target_path = join(data_dir, "stc_weibo_train_response")
    # sample_sources, sample_targets = sample_pairs(source_path, target_path, sample_size)
    # make_train_test_dev(sample_sources, sample_targets, ".")

    # filter_pos("../data/q2q_all.train", "../data/q2q_pos.train")

    # app="q2q_12w"
    #
    # main("../data/q2q_pos.train","../data/{}".format(app))

    cli()

