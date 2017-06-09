
#encoding=utf-8

import numpy as np
import tensorflow as tf

class SequenceTable:
  def __init__(self, data):
    # A TensorArray is required as the sequences don't have the same
    # length. Alternatively a FIFOQueue can be used.
    # Because the data is read more than once by the queue,
    # clear_after_read is set to False (but I can't confirm an effect).
    # Because the items has diffrent sequence lengths the infer_shape
    # is set to False. The shape is then restored in the .read method.
    self.table = tf.TensorArray(size=len(data),
                                dtype=data[0].dtype,
                                dynamic_size=False,
                                clear_after_read=False,
                                infer_shape=False)

    # initialize table
    for i, datum in enumerate(data):
      self.table = self.table.write(i, datum)

    # setup infered element shape
    self.element_shape = tf.TensorShape((None,) + data[0].shape[1:])

  def read(self, index):
    # read index from table and set infered shape
    read = self.table.read(index)
    read.set_shape(self.element_shape)
    return read


def shuffle_bucket_batch(input_length, tensors, shuffle=True, **kwargs):
  # bucket_by_sequence_length requires the input_length and tensors
  # arguments to be queues. Use a range_input_producer queue to shuffle
  # an index for sliceing the input_length and tensors laters.
  # This strategy is idendical to the one used in slice_input_producer.
  table_index = tf.train.range_input_producer(
    int(input_length.get_shape()[0]), shuffle=shuffle
  ).dequeue()

  # the first argument is the sequence length specifed in the input_length
  # I did not find a ue for it.
  _, batch_tensors = tf.contrib.training.bucket_by_sequence_length(
    input_length=tf.gather(input_length, table_index),
    tensors=[tensor.read(table_index) for tensor in tensors],
    **kwargs
  )

  return tuple(batch_tensors)



def test_main():
  # these values specify the length of the sequence and this controls how
  # the data is bucketed. The value is not required to be the acutal length,
  # which is also problematic when using pairs of sequences that have diffrent
  # length. In that case just specify a value that gives the best performance,
  # for example "the max length".
  length_table = tf.constant([2, 4, 3, 4, 3, 5], dtype=tf.int32)

  source_table = SequenceTable([
    np.asarray([3, 4], dtype=np.int32),
    np.asarray([2, 3, 4], dtype=np.int32),
    np.asarray([1, 3, 4], dtype=np.int32),
    np.asarray([5, 3, 4], dtype=np.int32),
    np.asarray([6, 3, 4], dtype=np.int32),
    np.asarray([3, 3, 3, 3, 3, 3], dtype=np.int32)
  ])

  target_table = SequenceTable([
    np.asarray([9], dtype=np.int32),
    np.asarray([9, 3, 4, 5], dtype=np.int32),
    np.asarray([9, 3, 4], dtype=np.int32),
    np.asarray([9, 3, 4, 6], dtype=np.int32),
    np.asarray([9, 3], dtype=np.int32),
    np.asarray([9, 3, 3, 3, 3, 3, 2], dtype=np.int32)
  ])

  source_batch, target_batch = shuffle_bucket_batch(
    length_table, [source_table, target_table],
    batch_size=2,
    # devices buckets into [len < 3, 3 <= len < 5, 5 <= len]
    bucket_boundaries=[1, 3, 5],
    # this will bad the source_batch and target_batch independently
    dynamic_pad=True,
    capacity=2
  )

  with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    for i in range(6):
      source, target = sess.run((source_batch, target_batch))
      print('source_output[{}]'.format(i))
      print(source)
      print('target_output[{}]'.format(i))
      print(target)
      print('')

    coord.request_stop()
    coord.join(threads)

if __name__ == "__main__":
  test_main()