#encoding=utf-8


import os
import sys
import codecs
import six
import shutil
import subprocess
import numpy as np
from os.path import join

import seq2seq

project_path = os.path.dirname(seq2seq.__path__[0])
generate_vocb_script_path = join(project_path, "bin/tools/generate_vocab.py")

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
        train_ratio = 0.8
        dev_ratio = 0.9
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

if __name__ == '__main__':

    # source_path = join(data_dir, "stc_weibo_train_post")
    # target_path = join(data_dir, "stc_weibo_train_response")
    # sample_sources, sample_targets = sample_pairs(source_path, target_path, sample_size)
    # make_train_test_dev(sample_sources, sample_targets, ".")

    sample_size = 500000
    sample_size = None
    all_pairs_file = "./question_gen/wenda_q2q.seg"
    app="question_gen_all"

    sample_sources, sample_targets = sample_ques_pairs(all_pairs_file, sample_size)
    make_train_test_dev(sample_sources, sample_targets, "./{app}".format(app=app))

    train_source_path = "./{app}/train/sources.txt".format(app=app)
    train_source_vocab_path = "./{app}/vocab/vocab.sources.txt".format(app=app)

    train_target_path = "./{app}/train/targets.txt".format(app=app)
    train_target_vocab_path = "./{app}/vocab/vocab.sources.txt".format(app=app)

    merge_path = "./{app}/train/merge_train_target.txt".format(app=app)
    shared_vocab_path = "./{app}/vocab/shared.vocab.txt".format(app=app)

    merge_files([train_source_path, train_target_path], merge_path)

    generate_vocb(generate_vocb_script_path, merge_path, shared_vocab_path)

    os.remove(merge_path)

    generate_vocb(generate_vocb_script_path, train_target_path,train_target_vocab_path)
