import tensorflow as tf
import numpy as np

def weight_variable(shape, name=None):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=name)

def bias_variable(shape, init=0.1, name=None):
	initial = tf.constant(init, shape=shape)
	return tf.Variable(initial, name=name)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x, name = None):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)

def create_model():
	model = tf.Graph()
	with model.as_default():
		#input
		_tiles = tf.placeholder(tf.float32, [None, 20, 10], name="tiles")
		_column = tf.placeholder(tf.float32, [None, 10], name="column")
		_current = tf.placeholder(tf.float32, [None, 2], name="current") # idx, next_idx
		keep_prob = tf.placeholder(tf.float32, name="kp")
		print("_tiles", _tiles)
		print("_current", _current)

		#layer 1
		_tiles_reshape = tf.reshape(_tiles, [-1, 20, 10, 1])
		W_conv1 = weight_variable([5,5,1,64], name="W_conv1")
		b_conv1 = bias_variable([64], name="b_conv1")
		h_conv1 = tf.nn.relu(conv2d(_tiles_reshape, W_conv1) + b_conv1)
		h_pool1 = max_pool_2x2(h_conv1, name="h_pool1")
		print("h_pool1", h_pool1)

		#layer 2 感觉第二次卷积会让图像太简单了，所以去掉
		# W_conv2 = weight_variable([3,3,32,64])
		# b_conv2 = bias_variable([64])
		# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
		# h_pool2 = max_pool_2x2(h_conv2, name="h_pool2")
		# print("h_pool2", h_pool2) # 5*3

		#layer fc1
		W_fc1 = weight_variable([10 * 5 * 64 + 2, 1024])
		b_fc1 = bias_variable([1024])
		h_pool_flat = tf.reshape(h_pool1, [-1, 10 * 5 * 64])
		h_fc1_input = tf.concat([h_pool_flat, _current], 1)
		h_fc1 = tf.nn.relu(tf.matmul(h_fc1_input, W_fc1) + b_fc1)
		print("h_fc1_input", h_fc1_input)
		print("h_fc1", h_fc1)

		#layer fc2
		W_fc2 = weight_variable([1024,256])
		b_fc2 = bias_variable([256])
		h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

		#drop out
		h_drop = tf.nn.dropout(h_fc2, keep_prob)
		print("h_drop", h_drop)

		#layer out x * 4 + r
		W_out_xr = weight_variable([256, 40])
		b_out_xr = bias_variable([40])

		# 这里如果使用softmax，那么最大值永远不会超过1，也就失去Q值得含义了
		#output = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2_xr) + b_fc2_xr, name="output")	# this is the Q of each action
		output = tf.add(tf.matmul(h_drop, W_out_xr), b_out_xr, name="output")	# this is the Q of each action
		print("output", output)

	return model

def model_2_dnn_layer(h_fc_input_rx, h_fc_input_rx_len):
	W_fc1_rx = weight_variable([h_fc_input_rx_len, 1024])
	b_fc1_rx = bias_variable([1024])
	h_fc1_rx = tf.nn.relu6(tf.matmul(h_fc_input_rx, W_fc1_rx) + b_fc1_rx)
	print("h_fc1_rx", h_fc1_rx)

	#layer fc2 for rx
	W_fc2_rx = weight_variable([1024,1024])
	b_fc2_rx = bias_variable([1024])
	h_fc2_rx = tf.nn.relu6(tf.matmul(h_fc1_rx, W_fc2_rx) + b_fc2_rx)
	print("h_fc2_rx", h_fc2_rx)

	#layer fc3 for rx
	W_fc3_rx = weight_variable([1024,1024])
	b_fc3_rx = bias_variable([1024])
	h_fc3_rx = tf.nn.relu6(tf.matmul(h_fc2_rx, W_fc3_rx) + b_fc3_rx)
	print("h_fc3_rx", h_fc3_rx)

	#out rx
	W_rx = weight_variable([1024, 10])
	b_rx = bias_variable([10], init=0.0)
	h_rx = tf.matmul(h_fc3_rx, W_rx) + b_rx
	print("h_rx", h_rx)
	return h_rx

def create_model_2():
	model = tf.Graph()
	with model.as_default():
		#input
		_tiles = tf.placeholder(tf.float32, [None, 20, 10], name="tiles")
		_column = tf.placeholder(tf.float32, [None, 10], name="column")
		_current = tf.placeholder(tf.float32, [None, 2], name="current") # idx, next_idx
		keep_prob = tf.placeholder(tf.float32, name="kp")
		print("_tiles", _tiles)
		print("_current", _current)

		#layer conv 1
		_tiles_reshape = tf.reshape(_tiles, [-1, 20, 10, 1])
		W_conv1 = weight_variable([5,5,1,64], name="W_conv1")
		b_conv1 = bias_variable([64], name="b_conv1")
		h_conv1 = tf.nn.relu(conv2d(_tiles_reshape, W_conv1) + b_conv1)
		h_pool1 = max_pool_2x2(h_conv1, name="h_pool1")
		print("h_pool1", h_pool1)
		#flat
		h_pool_flat_len = 10 * 5 * 64
		h_pool_flat = tf.reshape(h_pool1, [-1, h_pool_flat_len])
		print("h_pool_flat", h_pool_flat)

		# ====================================================================
		# 分别计算4中旋转之下位置的权重
		h_fc_input_rx = tf.concat([h_pool_flat, _column, _current], 1)
		h_fc_input_rx_len = h_pool_flat_len + 10 + 2
		# h_fc_input_rx = tf.concat([h_pool_flat, _current], 1)
		# h_fc_input_rx_len = h_pool_flat_len + 2
		h_r0 = model_2_dnn_layer(h_fc_input_rx, h_fc_input_rx_len)
		h_r1 = model_2_dnn_layer(h_fc_input_rx, h_fc_input_rx_len)
		h_r2 = model_2_dnn_layer(h_fc_input_rx, h_fc_input_rx_len)
		h_r3 = model_2_dnn_layer(h_fc_input_rx, h_fc_input_rx_len)

		# ==================================================================

		#output 将上述的四组权重合并
		output = tf.concat([h_r0, h_r1, h_r2, h_r3], 1, name="output")
		print("output", output)

	return model
