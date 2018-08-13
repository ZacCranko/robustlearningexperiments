import numpy as np
import tensorflow as tf

# Copyright 2018 Google LLC.
# SPDX-License-Identifier: Apache-2.0
def power_iterate_conv(layer, num_iter):
  """Perform power iteration for a convolutional layer."""
  assert isinstance(layer, tf.keras.layers.Conv2D)
  weights = layer.kernel
  strides = (1,) + layer.strides + (1,)
  padding = layer.padding.upper()
  
  with tf.variable_scope(None, default_name='power_iteration'):
    u_var = tf.get_variable(
       'u_conv', [1] + map(int, layer.output_shape[1:]),
       initializer=tf.random_normal_initializer(),
       trainable=False)
    u = u_var
    
    for _ in xrange(num_iter):
      v = tf.nn.conv2d_transpose(
         u, weights, [1] + map(int, layer.input_shape[1:]), strides, padding)
      v /= tf.sqrt(tf.maximum(2 * tf.nn.l2_loss(v), 1e-12))
      u = tf.nn.conv2d(v, weights, strides, padding)
      u /= tf.sqrt(tf.maximum(2 * tf.nn.l2_loss(u), 1e-12))
      
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.assign(u_var, u))

    u = tf.stop_gradient(u)
    v = tf.stop_gradient(v)
    return tf.reduce_sum(u * tf.nn.conv2d(v, weights, strides, padding))
  
def power_iterate_dense(layer, num_iter):
  """Perform power iteration for a fully connected layer."""
  assert isinstance(layer, tf.keras.layers.Dense)
  weights = layer.kernel
  output_shape, input_shape = weights.get_shape().as_list()

  with tf.variable_scope(None, default_name='power_iteration'):
    u_var = tf.get_variable(
       'u',  map(int, [output_shape]) + [1],
       initializer=tf.random_normal_initializer(),
       trainable=False)
    u = u_var

    for _ in xrange(num_iter):
      v = tf.matmul(weights, u, transpose_a=True)
      v /= tf.sqrt(tf.maximum(2 * tf.nn.l2_loss(v), 1e-12))
      u = tf.matmul(weights, v)
      u /= tf.sqrt(tf.maximum(2 * tf.nn.l2_loss(u), 1e-12))

    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.assign(u_var, u))

    u = tf.stop_gradient(u)
    v = tf.stop_gradient(v)
    return tf.reduce_sum(u * tf.matmul(weights, v))

def operator_norm(layer, ord = 2, **kwargs):
    """Compute operator norm for a Keras layer."""
    
  with tf.variable_scope(None, default_name='operator_norm'):
    if ord == 1:
      w = layer.kernel
      if isinstance(layer, tf.keras.layers.Conv2D):
        sum_w = tf.reduce_sum(tf.abs(w), [0, 1, 3])
      else:
        sum_w = tf.reduce_sum(tf.abs(w), 1)

      return tf.reduce_max(sum_w)

    elif ord == 2:
      num_iter = kwargs.get('num_iter', 5)
      if isinstance(layer, tf.keras.layers.Conv2D):
        return power_iterate_conv(layer, num_iter)
      else:
        return power_iterate_dense(layer, num_iter)

    elif ord == np.inf:
      w = layer.kernel
      if isinstance(layer, tf.keras.layers.Conv2D):
        sum_w = tf.reduce_sum(tf.abs(w), [0, 1, 2])
      else:
        sum_w = tf.reduce_sum(tf.abs(w), 0)

      return tf.reduce_max(sum_w)