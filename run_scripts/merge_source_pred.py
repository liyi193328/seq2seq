#encoding=utf-8


import codecs
def merge_source_pred(source_path, pred_path, save_path):
  source_lines =codecs.open(source_path,"r","utf-8").readlines()
  pred_lines = codecs.open(pred_path, "r", "utf-8").readlines()
  assert  len(source_lines) == len(pred_lines)
  f = codecs.open(save_path, "w", "utf-8")
  for i, source_line in enumerate(source_lines):
    newpairs = source_line + pred_lines[i] + "\n"
    f.write(newpairs)
  f.close()

if __name__ == "__main__":
  merge_source_pred(r"C:\Users\liyi1\Desktop\sources.txt", r"C:\Users\liyi1\Desktop\predictions.txt", r"C:\Users\liyi1\Desktop\source_pred.txt")
