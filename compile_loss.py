from keras.src import ops
from keras.src import tree
from keras.src import metrics as metrics_module
from keras.src.trainers import compile_utils
from keras.src.trainers.compile_utils import get_loss
from keras.src.utils.module_utils import optree


class CompileLoss(compile_utils.CompileLoss):
  def build(self, y_true, y_pred):
    loss = self._user_loss
    loss_weights = self._user_loss_weights
    output_names = self._get_y_pred_output_names(y_pred)
    inferred_output_names = output_names or self.output_names

    if isinstance(loss, dict):
      loss_keys = set(loss.keys())
      accessors = optree.tree_accessors(y_pred)

      flat_losses = []
      remaining_loss_paths = loss_keys.copy()

      def fn(a):
        path = '.'.join([str(e) for e in a.path])
        for key in remaining_loss_paths:
          if path.startswith(key):
            loss_accessor = optree.PyTreeAccessor(a[:len(key.split('.'))])
            flat_losses.append((loss_accessor, loss[key]))
            remaining_loss_paths.remove(key)
            break

      remaining_count = len(remaining_loss_paths)
      while remaining_loss_paths:
        optree.tree_map(fn, accessors)
        if remaining_count == len(remaining_loss_paths):
          raise KeyError(f"There are keys: {list(remaining_loss_paths)} in "
                           "the `loss` argument, but they can't be found in "
                           "the model's output (`y_pred`).")
        remaining_count = len(remaining_loss_paths)
    else:
      flat_losses = []
      # TODO

    # Get the real loss instances.
    flat_losses = [(accessor, get_loss(identifier, accessor(y_true), accessor(y_pred))) for accessor, identifier in flat_losses]

    # Add `Mean` metric to the tracker for each loss.
    if len(flat_losses) > 1:
      for accessor, _loss in flat_losses:
        if _loss is not None:
          if inferred_output_names is not None:
            name = '.'.join([str(e) for e in accessor.path])
          else:
            name = _loss.name
          name += "_loss"
          self._tracker.add_to_store("metrics", metrics_module.Mean(name=name))

    if loss_weights is None:
      flat_loss_weights = [None] * len(flat_losses)
    else:
      flat_loss_weights = tree.flatten(loss_weights)
      for loss_weight in flat_loss_weights:
        if not isinstance(loss_weight, (int, float, type(None))):
          raise TypeError("When providing the `loss_weights` argument, each "
                          "element should be a Python int, float (the weighting "
                          "coefficient corresponding to the loss for that "
                          "output) or `None`."
                          f"Received: loss_weights={loss_weights}")
      if len(flat_loss_weights) != len(flat_losses):
        raise ValueError(
          "When providing the `loss_weights` argument, it should "
          "have equal length of `loss` argument. "
          f"Received: loss_weights length={len(flat_loss_weights)}, "
          f"loss length={len(flat_losses)}")

    self.flat_losses = flat_losses
    self.flat_loss_weights = flat_loss_weights
    self.inferred_output_names = inferred_output_names
    self.built = True

  def call(self, y_true, y_pred, sample_weight=None):
    if not self.built:
      self.build(y_true, y_pred)

    if sample_weight is not None:
      # TODO
      pass
    else:
      sample_weight = [None for _ in self.flat_losses]

    # We need to add a dummy `None` if the model has only a single output.
    metrics = [None] if len(self.metrics) == 0 else self.metrics

    # Iterate all losses in flat form.
    loss_values = []
    for (accessor, loss_fn), loss_weight, sample_weight, metric in zip(self.flat_losses, self.flat_loss_weights, sample_weight, metrics):
      y_t, y_p = accessor(y_true), accessor(y_pred)

      if loss_fn:
        value = ops.cast(loss_fn(y_t, y_p, sample_weight), dtype=self.dtype)
        if loss_weight is not None:
          value = ops.multiply(value, loss_weight)
        loss_values.append(value)
        # Record individual losses.
        if metric:
          metric.update_state(value,
            sample_weight=tree.flatten(y_p)[0].shape[0])

    if loss_values:
      total_loss = sum(loss_values)
      return total_loss
    return None