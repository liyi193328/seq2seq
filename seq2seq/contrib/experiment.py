# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A patched tf.learn Experiment class to handle GPU memory
sharing issues.
"""

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.experiment import  _new_attr_context
from tensorflow.contrib.learn.python.learn import monitors

class Experiment(tf.contrib.learn.Experiment):
  """A patched tf.learn Experiment class to handle GPU memory
  sharing issues."""

  def __init__(self, train_steps_per_iteration=None, *args, **kwargs):
    super(Experiment, self).__init__(*args, **kwargs)
    self._train_steps_per_iteration = train_steps_per_iteration

  def _has_training_stopped(self, eval_result):
    """Determines whether the training has stopped."""
    if not eval_result:
      return False

    global_step = eval_result.get(tf.GraphKeys.GLOBAL_STEP)
    return global_step and self._train_steps and (
        global_step >= self._train_steps)

  def continuous_train_and_eval(self,
                                continuous_eval_predicate_fn=None):
    """Interleaves training and evaluation.

    The frequency of evaluation is controlled by the `train_steps_per_iteration`
    (via constructor). The model will be first trained for
    `train_steps_per_iteration`, and then be evaluated in turns.

    This differs from `train_and_evaluate` as follows:
      1. The procedure will have train and evaluation in turns. The model
      will be trained for a number of steps (usuallly smaller than `train_steps`
      if provided) and then be evaluated.  `train_and_evaluate` will train the
      model for `train_steps` (no small training iteraions).

      2. Due to the different approach this schedule takes, it leads to two
      differences in resource control. First, the resources (e.g., memory) used
      by training will be released before evaluation (`train_and_evaluate` takes
      double resources). Second, more checkpoints will be saved as a checkpoint
      is generated at the end of each small trainning iteration.

    Args:
      continuous_eval_predicate_fn: A predicate function determining whether to
        continue after each iteration. `predicate_fn` takes the evaluation
        results as its arguments. At the beginning of evaluation, the passed
        eval results will be None so it's expected that the predicate function
        handles that gracefully. When `predicate_fn` is not specified, this will
        run in an infinite loop or exit when global_step reaches `train_steps`.

    Returns:
      A tuple of the result of the `evaluate` call to the `Estimator` and the
      export results using the specified `ExportStrategy`.

    Raises:
      ValueError: if `continuous_eval_predicate_fn` is neither None nor
        callable.
    """

    if (continuous_eval_predicate_fn is not None and
        not callable(continuous_eval_predicate_fn)):
      raise ValueError(
          "`continuous_eval_predicate_fn` must be a callable, or None.")

    eval_result = None

    # Set the default value for train_steps_per_iteration, which will be
    # overriden by other settings.
    train_steps_per_iteration = 1000
    if self._train_steps_per_iteration is not None:
      train_steps_per_iteration = self._train_steps_per_iteration
    elif self._train_steps is not None:
      # train_steps_per_iteration = int(self._train_steps / 10)
      train_steps_per_iteration = min(
          self._min_eval_frequency, self._train_steps)

    while (not continuous_eval_predicate_fn or
           continuous_eval_predicate_fn(eval_result)):

      if self._has_training_stopped(eval_result):
        # Exits once max steps of training is satisfied.
        tf.logging.info("Stop training model as max steps reached")
        break

      tf.logging.info("Training model for %s steps", train_steps_per_iteration)
      self.train(delay_secs=0)

      # self._estimator.fit(
      #     input_fn=self._train_input_fn,
      #     steps=train_steps_per_iteration,
      #     monitors=self._train_monitors)

      tf.logging.info("Evaluating model now.")
      eval_result = self._estimator.evaluate(
          input_fn=self._eval_input_fn,
          steps=self._eval_steps,
          metrics=self._eval_metrics,
          name="one_pass",
          hooks=self._eval_hooks)

    return eval_result, self._maybe_export(eval_result)


  def dis_train_and_evaluate(self):
    """Interleaves training and evaluation.

    The frequency of evaluation is controlled by the contructor arg
    `min_eval_frequency`. When this parameter is None or 0, evaluation happens
    only after training has completed. Note that evaluation cannot happen
    more frequently than checkpoints are taken. If no new snapshots are
    available when evaluation is supposed to occur, then evaluation doesn't
    happen for another `min_eval_frequency` steps (assuming a checkpoint is
    available at that point). Thus, settings `min_eval_frequency` to 1 means
    that the model will be evaluated everytime there is a new checkpoint.

    This is particular useful for a "Master" task in the cloud, whose
    responsibility it is to take checkpoints, evaluate those checkpoints,
    and write out summaries. Participating in training as the supervisor
    allows such a task to accomplish the first and last items, while
    performing evaluation allows for the second.

    Returns:
      The result of the `evaluate` call to the `Estimator`.
    """
    # The directory to which evaluation summaries are written are determined
    # by adding a suffix to 'eval'; that suffix is the 'name' parameter to
    # the various evaluate(...) methods. By setting it to None, we force
    # the directory name to simply be 'eval'.
    eval_dir_suffix = "one_pass"

    # We set every_n_steps to 1, but evaluation only occurs when a new
    # snapshot is available. If, by the time we finish evaluation
    # there is a new snapshot, then we just evaluate again. Otherwise,
    # we keep training until one becomes available.
    with _new_attr_context(self, "_train_monitors"):
      self._train_monitors = self._train_monitors or []
      if self._min_eval_frequency:
        self._train_monitors += [monitors.ValidationMonitor(
            input_fn=self._eval_input_fn, eval_steps=self._eval_steps,
            metrics=self._eval_metrics, every_n_steps=self._min_eval_frequency,
            name=eval_dir_suffix,
        )]
      self.train(delay_secs=0)

    tf.logging.info("Evaluating model now.")

    eval_result = self._estimator.evaluate(
        input_fn=self._eval_input_fn,
        steps=self._eval_steps,
        metrics=self._eval_metrics,
        name="one_pass",
        hooks=self._eval_hooks)

    return eval_result, self._maybe_export(eval_result)