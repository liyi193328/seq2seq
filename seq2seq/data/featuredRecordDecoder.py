#encoding=utf-8

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.contrib.slim.python.slim.data import data_decoder
from seq2seq.data import split_tokens_decoder

def merge_dict(d1, d2):
  d = {}
  for k in d1:
    d[k] = d1[k]
  for k in d2:
    assert k not in d1, (d1.keys(), d2.keys())
    d[k] = d2[k]
  return d

class FeaturedTFExampleDecoder(data_decoder.DataDecoder):
  """A decoder for TensorFlow Examples.

  Decoding Example proto buffers is comprised of two stages: (1) Example parsing
  and (2) tensor manipulation.

  In the first stage, the tf.parse_example function is called with a list of
  FixedLenFeatures and SparseLenFeatures. These instances tell TF how to parse
  the example. The output of this stage is a set of tensors.

  In the second stage, the resulting tensors are manipulated to provide the
  requested 'item' tensors.

  To perform this decoding operation, an ExampleDecoder is given a list of
  ItemHandlers. Each ItemHandler indicates the set of features for stage 1 and
  contains the instructions for post_processing its tensors for stage 2.
  """

  def __init__(self, source_keys_to_tensor, target_keys_to_tensor=None, items_to_handlers=None):
    """Constructs the decoder.

    Args:
      source_keys_to_tensor: a dictionary from TF-Example keys to either
        tf.VarLenFeature or tf.FixedLenFeature instances. See tensorflow's
        parsing_ops.py.
      target_keys_to_tensor: None mean infer mode
    """
    self._source_keys_to_tensor = source_keys_to_tensor
    self._source_feature_keys = list(source_keys_to_tensor.keys())
    self._target_keys_to_tensor = target_keys_to_tensor
    self._target_feature_keys = []
    if target_keys_to_tensor is not None:
      self._target_feature_keys = list(target_keys_to_tensor.keys())
    self._keys_to_features = source_keys_to_tensor.copy()
    for k in target_keys_to_tensor:
      self._keys_to_features[k] = target_keys_to_tensor[k]

    if items_to_handlers is not None:
      self._items = items_to_handlers.keys()
    else:
      self._items = self._source_feature_keys + self._target_feature_keys

    self._items.extend(["source_len", "target_len"])

  def list_items(self):
    """See base class."""
    return self._items

  def decode(self, serialized_example, items=None):
    """Decodes the given serialized TF-example.

    Args:
      serialized_example: a serialized TF-example tensor.
      items: the list of items to decode. These must be a subset of the item
        keys in self._items_to_handlers. If `items` is left as None, then all
        of the items in self._items_to_handlers are decoded.

    Returns:
      the decoded items, a list of tensor.
    """

    example = parsing_ops.parse_single_example(serialized_example,
                                               self._keys_to_features)

    def get_feature_tensor(example, keys_to_features):
      features = {}
      # Reshape non-sparse elements just once:
      for k in keys_to_features:
        v = keys_to_features[k]
        if isinstance(v, parsing_ops.FixedLenFeature):
          example[k] = array_ops.reshape(example[k], v.shape)
        if isinstance(example[k], tf.SparseTensor):
          example[k] = example[k].values
        if example[k].dtype is not tf.int64:
          tokens = tf.string_split(example[k], delimiter=" ").values
          features[k] = tokens
        else:
          features[k] = example[k]
      return features

    source_feature_tensors = get_feature_tensor(example, self._source_keys_to_tensor)
    source_feature_tensors["source_len"] = tf.size(source_feature_tensors["source_tokens"],out_type=tf.int64)

    target_feature_tensors = {}
    have_target = False
    if self._target_feature_keys[0] in example:
      have_target = True
      target_feature_tensors = get_feature_tensor(example, self._target_keys_to_tensor)
      target_feature_tensors["target_len"] = tf.size(target_feature_tensors["target_tokens"], out_type=tf.int64)

    all_features = merge_dict(source_feature_tensors, target_feature_tensors)

    outputs = [ all_features[v] for v in self._items ]

    return outputs