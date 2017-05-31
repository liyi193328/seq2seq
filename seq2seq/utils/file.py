#encoding=utf-8

import shutil
import os
import sys

def copy_checkpoint_files(path_prefix, dst_dir,add_name="best"):
  if os.path.exists(dst_dir) == False:
    os.makedirs(dst_dir)
  path_prefix = os.path.abspath(path_prefix)
  prefix = os.path.basename(path_prefix)
  source_dir = os.path.dirname(path_prefix)

  checkpoint_files = [os.path.join(source_dir,v) for v in os.listdir(source_dir) if path_prefix in v]
  for checkpoint_file in checkpoint_files:
    name = os.path.basename(checkpoint_file)
    dst_name = "{}.{}".format(add_name,name)
    dst_path = os.path.join(dst_dir,dst_name)
    shutil.copy(checkpoint_file,dst_path)