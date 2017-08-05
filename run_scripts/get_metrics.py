#encoding=utf-8

import os
import logging
import codecs
import subprocess
import rouge
import json
import click
import tempfile
from collections import OrderedDict

def get_bleu_info(ref_path, pred_path):

  print("######cal belu######")
  metrics_dir = os.path.dirname(os.path.realpath(__file__))
  bin_dir = os.path.abspath(os.path.join(metrics_dir, "..","bin"))
  multi_bleu_path = os.path.join(bin_dir, "tools/multi-bleu.perl")
  print("using local script in {} to get bleu score ".format(multi_bleu_path))

  # Calculate BLEU using multi-bleu script
  with codecs.open(pred_path, "r", "utf-8") as read_pred:
    bleu_cmd = [multi_bleu_path]
    bleu_cmd += [ref_path]
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

def get_rouge(ref_path, pred_path):
  """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
  print("###########cal rouge###############")
  source_lines = [line.strip() for line in codecs.open(ref_path, "r", "utf-8").readlines()]
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

def copy_model_post_fn(line):
  """unique sentence end !!!! => !
  :param line: 
  :return: 
  """
  tokens = line.strip().split(" ")
  if len(tokens) == 0:
    return ""
  else:
    last_token = tokens[-1]
    new_last_token = []
    char_set = set()
    for char in last_token:
      if char not in new_last_token:
        new_last_token.append(char)
    new_last_token = "".join(new_last_token)
    tokens[-1] = new_last_token
  return " ".join(tokens)

@click.command()
@click.argument("pred_path")
@click.argument("ref_path")
@click.argument("format")
@click.argument("result_path")
def main(pred_path, ref_path, format, result_path):
  logging.warn("source_path and pred_path must with line one by one")
  pred_post_name = os.path.basename(pred_path).split(".")[0] + ".extract.pred"
  all_result_path_name = pred_post_name.replace(".extract.pred", ".all.result")
  extract_pred_path = os.path.join(os.path.join(os.path.dirname(result_path), pred_post_name))
  all_result_path = os.path.join(os.path.join(os.path.dirname(result_path), all_result_path_name))
  pred_fout = codecs.open(extract_pred_path, "w", "utf-8")
  all_result_fout = codecs.open(all_result_path, "w", "utf-8")
  fin = codecs.open(pred_path, "r", "utf-8")
  ref_fin = codecs.open(ref_path, "r", "utf-8")
  if format == "source_beam_search":
    while True:
      line = fin.readline()
      ref_line = ref_fin.readline()
      all_result_fout.write("source:\n{}\nref:\n{}\npred:\n".format(line.strip(), ref_line.strip()))
      if not line:
        break
      beam_search_lines = []
      while True:
        b = fin.readline()
        if b == "" or b.strip() == "":
          break
        all_result_fout.write(b)
        beam_search_lines.append(b)
      assert len(beam_search_lines) > 0, (line,beam_search_lines)
      pred_fout.write(beam_search_lines[0])
      all_result_fout.write("\n")
  elif format == "copy_pred":
    while True:
      line = fin.readline()
      if not line:
        break
      new_line = copy_model_post_fn(line)
      ref_line = ref_fin.readline()
      all_result_fout.write("source:\n{}\nref:{}\npred:".format(line.strip(), ref_line.strip()))
      pred_fout.write(new_line.strip() + "\n")
      all_result_fout.write(new_line.strip() + "\n")
      all_result_fout.write("\n")
  else:
    pred_fout.close()
    fin.close()
    all_result_fout.close()
    raise ValueError("{} not in copy")
  pred_fout.close()
  fin.close()
  all_result_fout.close()
  print("pred extract from {} to {}".format(pred_path, extract_pred_path))
  print("source pred[post] write to {}".format(all_result_path))
  bleu_out = get_bleu_info(ref_path, extract_pred_path)
  rouge_score = get_rouge(ref_path, extract_pred_path)
  result = {}
  result["bleu"] = bleu_out
  result["rouge"] = rouge_score
  json.dump(result, codecs.open(result_path, "w", "utf-8"), indent=2)
  print("write result to {}".format(result_path))
  return

if __name__ == "__main__":
  main()

