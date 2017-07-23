# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.data import data_provider
from tensorflow.contrib.slim.python.slim.data import parallel_reader

class FeaturedDataProvider(data_provider.DataProvider):

  def __init__(self,
               dataset,
               num_readers=1,
               reader_kwargs=None,
               shuffle=True,
               num_epochs=None,
               common_queue_capacity=256,
               common_queue_min=128,
               record_key='record_key',
               seed=None,
               scope=None):
    """Creates a DatasetDataProvider.

    Args:
      dataset: An instance of the Dataset class.
      num_readers: The number of parallel readers to use.
      reader_kwargs: An optional dict of kwargs for the reader.
      shuffle: Whether to shuffle the data sources and common queue when
        reading.
      num_epochs: The number of times each data source is read. If left as None,
        the data will be cycled through indefinitely.
      common_queue_capacity: The capacity of the common queue.
      common_queue_min: The minimum number of elements in the common queue after
        a dequeue.
      record_key: The item name to use for the dataset record keys in the
        provided tensors.
      seed: The seed to use if shuffling.
      scope: Optional name scope for the ops.
    Raises:
      ValueError: If `record_key` matches one of the items in the dataset.
    """
    key, data = parallel_reader.parallel_read(
        dataset.data_sources,
        reader_class=dataset.reader,
        num_epochs=num_epochs,
        num_readers=num_readers,
        reader_kwargs=reader_kwargs,
        shuffle=shuffle,
        capacity=common_queue_capacity,
        min_after_dequeue=common_queue_min,
        seed=seed,
        scope=scope)

    items = dataset.decoder.list_items()
    tensors = dataset.decoder.decode(data, items)

    super(FeaturedDataProvider, self).__init__(
        items_to_tensors=dict(zip(items, tensors)),
        num_samples=dataset.num_samples)