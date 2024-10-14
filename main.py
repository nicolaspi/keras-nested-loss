import keras
import numpy as np
import tensorflow as tf
from keras.src import tree

from losses import LossFunctionWrapper
from compile_loss import CompileLoss

keras.config.disable_traceback_filtering()

# Generate some dummy data
np.random.seed(0)
data_shape = (10, 2)
X = np.random.rand(*data_shape)
y1 = np.random.rand(*data_shape)
y2 = np.random.rand(*data_shape)
y3 = np.random.rand(*data_shape)

train_set = tf.data.Dataset.from_tensor_slices((X, {'a': y1, 'b': {'c': (y1, y2), 'd': y3 }})).batch(2)

# Define a custom loss function for multiple outputs
def loss_fn(y_true, y_pred):
    flat_y_pred, flat_y_true = tree.flatten(y_pred), tree.flatten(y_true)
    diff = tf.constant(0.0)
    for y_p, y_t in zip(flat_y_pred, flat_y_true):
      diff += keras.losses.mean_absolute_error(y_t, y_p)
    return diff

# Define the model
inputs = keras.Input(shape=data_shape[1:])
x = keras.layers.Dense(64, activation='relu')(inputs)
x = keras.layers.Dense(32, activation='relu')(x)
y1 = keras.layers.Dense(data_shape[-1], name='y1')(x)
y2 = keras.layers.Dense(data_shape[-1], name='y2')(x)
y3 = keras.layers.Dense(data_shape[-1], name='y3')(x)

model = keras.Model(inputs=inputs, outputs={'a': y1, 'b': {'c': (y1, y2), 'd': y3 }})

loss = {'a': loss_fn, 'b': loss_fn, 'b.c': loss_fn, 'b.d': loss_fn }

# Compile the model
model.compile(optimizer='adam', loss=loss, jit_compile=False)

# Patching keras with POC's logic
keras.src.losses.LossFunctionWrapper = LossFunctionWrapper
model._compile_loss = CompileLoss(
  model._compile_loss._user_loss,
  model._compile_loss._user_loss_weights,
  output_names=model._compile_loss.output_names
  )

# Train the model
history = model.fit(
    train_set,
    epochs=100,
    batch_size=2,
    verbose=1
)
