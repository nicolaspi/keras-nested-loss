from keras.src.losses import LossFunctionWrapper as KerasLossFunctionWrapper
from keras.src.losses.loss import squeeze_or_expand_to_same_rank
from keras.src.utils.module_utils import optree


class LossFunctionWrapper(KerasLossFunctionWrapper):
  def call(self, y_true, y_pred):
    y_true, y_pred = optree.tree_transpose_map(squeeze_or_expand_to_same_rank, y_true, y_pred)
    return self.fn(y_true, y_pred, **self._fn_kwargs)

