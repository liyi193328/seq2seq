#encoding=utf-8

import os
import logging
import codecs
import subprocess
import rouge
import json
import click
from collections import OrderedDict

def get_bleu_info(source_path, pred_path):

  print("######cal belu######")
  metrics_dir = os.path.dirname(os.path.realpath(__file__))
  bin_dir = os.path.abspath(os.path.join(metrics_dir, "..","bin"))
  multi_bleu_path = os.path.join(bin_dir, "tools/multi-bleu.perl")
  print("using local script in {} to get bleu score ".format(multi_bleu_path))

  # Calculate BLEU using multi-bleu script
  with codecs.open(pred_path, "r", "utf-8") as read_pred:
    bleu_cmd = [multi_bleu_path]
    bleu_cmd += [source_path]
    try:
      bleu_out = subprocess.check_output(
          bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
      bleu_out = bleu_out.decode("utf-8")
      # bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
      # bleu_score = float(bleu_score)
    except subprocess.CalledProcessError as error:
      if error.output is not None:
        logging.warning("multi-bleu.perl script returned non-zero exit code")
        logging.warning(error.output)
      bleu_out = "0.0, multi-bleu.perl script returned non-zero exit code"

  print(bleu_out)
  print("#############end bleu#########\n")
  return bleu_out

def get_rouge(source_path, pred_path):
  """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
  print("###########cal rouge###############")
  source_lines = [line.strip() for line in codecs.open(source_path, "r", "utf-8").readlines()]
  pred_lines = [ pred.strip() for pred in codecs.open(pred_path, "r", "utf-8").readlines() ]
  assert len(source_lines) == len(pred_lines)
  rou = rouge.Rouge()
  scores = rou.get_scores(pred_lines, source_lines)
  ave_scores = rou.get_scores(pred_lines, source_lines, avg=True)
  result = OrderedDict()
  result["ave_scores"] = ave_scores
  result["detail_scores"] = scores
  print(ave_scores)
  print("############end rouge#############\n")
  return result

@click.command()
@click.argument("source_path")
@click.argument("pred_path")
@click.argument("result_path")
def main(source_path, pred_path, result_path):
  logging.warn("source_path and pred_path must with line one by one")
  bleu_out = get_bleu_info(source_path, pred_path)
  rouge_score = get_rouge(source_path, pred_path)
  result = {}
  result["bleu"] = bleu_out
  result["rouge"] = rouge_score
  json.dump(result, codecs.open(result_path, "w", "utf-8"), indent=2)
  print("write result to {}".format(result_path))
  return

if __name__ == "__main__":
  main()

