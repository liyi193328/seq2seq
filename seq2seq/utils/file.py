#encoding=utf-8

import traceback
import shutil
import os
import sys

def cancel_prev_checkpoint_files(source_dir, add_name="best",keep_last_steps=1):
  step_paths = {}
  for i, v in enumerate(os.listdir(source_dir)):
    if v.startswith(add_name+"."):
      try:
        step = int(v.split("-")[1].split(".")[0])
        if step not in step_paths:
          step_paths[step] = []
        step_paths[step].append(os.path.join(source_dir, v))
      except:
        continue
  sort_keys = sorted(step_paths.keys())
  if len(sort_keys) <= keep_last_steps:
    return
  for i in range(len(sort_keys)-keep_last_steps-1, -1, -1):
    step = sort_keys[i]
    for path in step_paths[step]:
      os.remove(path)

def copy_checkpoint_files(path_prefix, dst_dir,add_name="best", keep_last_steps=1):
  if os.path.exists(dst_dir) == False:
    os.makedirs(dst_dir)
  path_prefix = os.path.abspath(path_prefix)
  prefix = os.path.basename(path_prefix)
  source_dir = os.path.dirname(path_prefix)

  checkpoint_files = [os.path.join(source_dir,v) for v in os.listdir(source_dir) if prefix in v]
  for checkpoint_file in checkpoint_files:
    name = os.path.basename(checkpoint_file)
    dst_name = "{}.{}".format(add_name,name)
    dst_path = os.path.join(dst_dir,dst_name)
    shutil.copy(checkpoint_file,dst_path)
  try:
    cancel_prev_checkpoint_files(source_dir, add_name=add_name, keep_last_steps=keep_last_steps)
  except:
    traceback.print_exc()

if __name__ == "__main__":
  cancel_prev_checkpoint_files(r"E:\active_project\test_tf\test_checkpoints")



