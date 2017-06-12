#encoding=utf-8

import sys
if sys.version_info[0] == 2:
  reload(sys)
  sys.setdefaultencoding("utf-8")

import os
import codecs
import json

def filter_low_sim_from_json(json_path, sim_threshold=0.95):
  data = json.load()
