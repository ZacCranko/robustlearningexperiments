import numpy as np
import tensorflow as tf

def tf_assert_almost_equal(actual, desired, **kwargs):
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    _actual  = actual.eval()
    _desired = desired.eval()

  return np.testing.assert_almost_equal(_actual, _desired, **kwargs)

def conv_matrix(layer):
  """Build the matrix associated with the convolution operation."""
  assert isinstance(layer, tf.keras.layers.Conv2D)
  with tf.variable_scope(None, default_name='build_conv_matrix'):
    weights = layer.kernel
    strides = (1,) + layer.strides + (1,)
    padding = layer.padding.upper()
    in_h,  in_w,  in_ch  = layer.input_shape[1:4]
    out_h, out_w, out_ch = layer.output_shape[1:4]

    id_mx     = tf.reshape(tf.eye(in_h*in_w*in_ch), 
                            (in_h*in_w*in_ch, in_h, in_w, in_ch))
    conv_mx_t = tf.reshape(tf.nn.conv2d(id_mx, weights, strides, padding), 
                            (in_h*in_w*in_ch, out_h*out_w*out_ch))
    return tf.transpose(conv_mx_t)

model = tf.keras.Sequential()
conv1 = tf.keras.layers.Conv2D(32, 5, 1, padding='SAME',
                               input_shape=(28, 28, 1))
model.add(conv1)
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(2, 2, padding='SAME'))

conv2 = tf.keras.layers.Conv2D(64, 5, 1, padding='SAME')
model.add(conv2)

dense1 = tf.keras.layers.Dense(1024)
model.add(dense1)
model.add(tf.keras.layers.Activation('relu'))

dense2 = tf.keras.layers.Dense(10)
model.add(dense2)

num_iter = 200

# test dense layers
for layer in ['dense1', 'dense2']:
  _layer = eval(layer)
  
  op = "1_norm"
  # axis = 1 here since _layer.kernel is stored transposed for dense layers
  inf_opn_mx = tf.reduce_max(tf.reduce_sum(tf.abs(_layer.kernel), axis = 1))
  tf_assert_almost_equal(operator_norm(_layer, 1), inf_opn_mx, err_msg = "%s(%s)"%(op,layer), decimal = 1)
  
  op = "inf_norm"
  # axis = 0 here since _layer.kernel is stored transposed for dense layers
  inf_opn_mx = tf.reduce_max(tf.reduce_sum(tf.abs(_layer.kernel), axis = 0))
  tf_assert_almost_equal(operator_norm(_layer, np.inf), inf_opn_mx, err_msg = "%s(%s)"%(op,layer), decimal = 1)
  
  op = "spectral_norm"
  spec_pow  = operator_norm(_layer, 2, num_iter = num_iter)
  spec_svd  = tf.svd(_layer.kernel, compute_uv=False)
  tf_assert_almost_equal(spec_pow, spec_svd[0], decimal = 2, err_msg = "%s(%s)"%(op,layer))

# test conv layers
for layer in ['conv1', 'conv2']:
  _layer = eval(layer)
  
  op = "1_norm"
  conv_mx = conv_matrix(_layer)
  desired = tf.reduce_max(tf.reduce_sum(tf.abs(conv_mx), axis = 0))
  tf_assert_almost_equal(operator_norm(_layer, 1), desired, err_msg = "%s(%s)"%(op,layer), decimal = 1)
  
  op = "inf_norm"
  conv_mx = conv_matrix(_layer)
  desired = tf.reduce_max(tf.reduce_sum(tf.abs(conv_mx), axis = 1))
  tf_assert_almost_equal(operator_norm(_layer, np.inf), desired, err_msg = "%s(%s)"%(op,layer), decimal = 1)
  
  op = "spectral_norm"
  spec_pow  = operator_norm(_layer, 2, num_iter = num_iter)
  spec_svd  = tf.svd(conv_matrix(_layer), compute_uv=False)
  tf_assert_almost_equal(spec_pow, spec_svd[0], err_msg = "%s(%s)"%(op,layer), decimal = 2)