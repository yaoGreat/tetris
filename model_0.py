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


# 模型2，将4个方向分成四个子模型，解决了T型问题
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

		h_r0 = model_2_dnn_layer(h_fc_input_rx, h_fc_input_rx_len)
		h_r1 = model_2_dnn_layer(h_fc_input_rx, h_fc_input_rx_len)
		h_r2 = model_2_dnn_layer(h_fc_input_rx, h_fc_input_rx_len)
		h_r3 = model_2_dnn_layer(h_fc_input_rx, h_fc_input_rx_len)

		# ==================================================================

		#output 将上述的四组权重合并
		output = tf.concat([h_r0, h_r1, h_r2, h_r3], 1, name="output")
		print("output", output)

	return model

# 模型3，将7个形状和4个方向分别设置一个DNN，很难训练，速度非常慢
def model_3_dnn_layer_light(h_fc_input_rx, h_fc_input_rx_len):
	W_fc1_rx = weight_variable([h_fc_input_rx_len, 1024])
	b_fc1_rx = bias_variable([1024])
	h_fc1_rx = tf.nn.relu6(tf.matmul(h_fc_input_rx, W_fc1_rx) + b_fc1_rx)
	#print("h_fc1_rx", h_fc1_rx)

	W_fc2_rx = weight_variable([1024, 512])
	b_fc2_rx = bias_variable([512])
	h_fc2_rx = tf.nn.relu6(tf.matmul(h_fc1_rx, W_fc2_rx) + b_fc2_rx)

	#out rx
	W_rx = weight_variable([512, 10])
	b_rx = bias_variable([10], init=0.0)
	h_rx = tf.matmul(h_fc2_rx, W_rx) + b_rx
	#print("h_rx", h_rx)
	return h_rx

def model_3_block_dnn_layer(h_fc_input_rx, h_fc_input_rx_len):
	h_r0 = model_3_dnn_layer_light(h_fc_input_rx, h_fc_input_rx_len)
	h_r1 = model_3_dnn_layer_light(h_fc_input_rx, h_fc_input_rx_len)
	h_r2 = model_3_dnn_layer_light(h_fc_input_rx, h_fc_input_rx_len)
	h_r3 = model_3_dnn_layer_light(h_fc_input_rx, h_fc_input_rx_len)
	return tf.concat([h_r0, h_r1, h_r2, h_r3], 1)

def create_model_3():
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
		# 根据7中方块分别计算，然后根据current来选择
		h_fc_input_rx = tf.concat([h_pool_flat, _column, _current], 1)
		h_fc_input_rx_len = h_pool_flat_len + 10 + 2

		h_b0 = model_3_block_dnn_layer(h_fc_input_rx, h_fc_input_rx_len)
		h_b1 = model_3_block_dnn_layer(h_fc_input_rx, h_fc_input_rx_len)
		h_b2 = model_3_block_dnn_layer(h_fc_input_rx, h_fc_input_rx_len)
		h_b3 = model_3_block_dnn_layer(h_fc_input_rx, h_fc_input_rx_len)
		h_b4 = model_3_block_dnn_layer(h_fc_input_rx, h_fc_input_rx_len)
		h_b5 = model_3_block_dnn_layer(h_fc_input_rx, h_fc_input_rx_len)
		h_b6 = model_3_block_dnn_layer(h_fc_input_rx, h_fc_input_rx_len)

		_block = tf.reshape(tf.slice(_current, begin=[0, 0], size=[-1,1]), [-1])
		print("_block", _block)
		_block_onehot = tf.one_hot(tf.cast(_block, tf.int32), 7)
		print("_block_onehot", _block_onehot)

		# _pack_bx = tf.stack([h_b0, h_b1, h_b2, h_b3, h_b4, h_b5, h_b6], 1)
		# print("_pack_bx", _pack_bx)

		multiples = [1, 40]
		h_s0 = h_b0 * tf.tile(tf.slice(_block_onehot, begin=[0, 0], size=[-1,1]), multiples)
		h_s1 = h_b1 * tf.tile(tf.slice(_block_onehot, begin=[0, 1], size=[-1,1]), multiples)
		h_s2 = h_b2 * tf.tile(tf.slice(_block_onehot, begin=[0, 2], size=[-1,1]), multiples)
		h_s3 = h_b3 * tf.tile(tf.slice(_block_onehot, begin=[0, 3], size=[-1,1]), multiples)
		h_s4 = h_b4 * tf.tile(tf.slice(_block_onehot, begin=[0, 4], size=[-1,1]), multiples)
		h_s5 = h_b5 * tf.tile(tf.slice(_block_onehot, begin=[0, 5], size=[-1,1]), multiples)
		h_s6 = h_b6 * tf.tile(tf.slice(_block_onehot, begin=[0, 6], size=[-1,1]), multiples)
		print("h_s0", h_s0)
		# ==================================================================

		#output 将上述的四组权重合并
		output = tf.add_n([h_s0, h_s1, h_s2, h_s3, h_s4, h_s5, h_s6], name="output")
		print("output", output)

	return model


# 模型4，将旋转和方块的组合用one_hot向量来表示，合并到一个全连接网络之中，看是否能提高训练速度
def max_pool_2x1(x, name = None):
	return tf.nn.max_pool(x, ksize=[1,2,1,1], strides=[1,2,1,1], padding='SAME', name=name)


def create_model_4():
	model = tf.Graph()
	with model.as_default():
		#input
		_tiles = tf.placeholder(tf.float32, [None, 20, 10], name="tiles")
		_column = tf.placeholder(tf.float32, [None, 10], name="column")
		_current = tf.placeholder(tf.float32, [None, 2], name="current") # idx, next_idx
		keep_prob = tf.placeholder(tf.float32, name="kp")
		print("_tiles", _tiles)
		print("_current", _current)

		_tiles_reshape = tf.reshape(_tiles, [-1, 20, 10, 1])

		#layer conv 5 x 5
		# W_conv5 = weight_variable([5,5,1,64], name="W_conv5")
		# b_conv5 = bias_variable([64], name="b_conv5")
		# h_conv5 = tf.nn.relu(conv2d(_tiles_reshape, W_conv5) + b_conv5)
		# h_pool5 = max_pool_2x1(h_conv5, name="h_pool5")
		# h_pool5_flat_len = 10 * 10 * 64
		# h_pool5_flat = tf.reshape(h_pool5, [-1, h_pool5_flat_len])
		# print("h_pool5", h_pool5)
		# print("h_pool5_flat", h_pool5_flat)

		#layer conv 3 x 3
		W_conv3 = weight_variable([3,3,1,64], name="W_conv3")
		b_conv3 = bias_variable([64], name="b_conv3")
		h_conv3 = tf.nn.relu(conv2d(_tiles_reshape, W_conv3) + b_conv3)
		h_pool3 = max_pool_2x1(h_conv3, name="h_pool3")
		h_pool3_flat_len = 10 * 10 * 64
		h_pool3_flat = tf.reshape(h_pool3, [-1, h_pool3_flat_len])
		print("h_pool3", h_pool3)
		print("h_pool_flat", h_pool3_flat)

		# 当前方块和下一个方块的onehot
		_block = tf.reshape(tf.slice(_current, begin=[0, 0], size=[-1,1]), [-1])
		_block_onehot = tf.one_hot(tf.cast(_block, tf.int32), 7, on_value=5.0) #让当前方块的索引更明显一点
		_nextblock = tf.reshape(tf.slice(_current, begin=[0, 1], size=[-1,1]), [-1])
		_nextblock_onehot = tf.one_hot(tf.cast(_nextblock, tf.int32), 7)

		# 每一列高度的onehot
		# _c0_onehot = tf.one_hot(tf.cast(tf.reshape(tf.slice(_column, begin=[0, 0], size=[-1,1]), [-1]), tf.int32), 20)		
		# _c1_onehot = tf.one_hot(tf.cast(tf.reshape(tf.slice(_column, begin=[0, 1], size=[-1,1]), [-1]), tf.int32), 20)		
		# _c2_onehot = tf.one_hot(tf.cast(tf.reshape(tf.slice(_column, begin=[0, 2], size=[-1,1]), [-1]), tf.int32), 20)		
		# _c3_onehot = tf.one_hot(tf.cast(tf.reshape(tf.slice(_column, begin=[0, 3], size=[-1,1]), [-1]), tf.int32), 20)		
		# _c4_onehot = tf.one_hot(tf.cast(tf.reshape(tf.slice(_column, begin=[0, 4], size=[-1,1]), [-1]), tf.int32), 20)		
		# _c5_onehot = tf.one_hot(tf.cast(tf.reshape(tf.slice(_column, begin=[0, 5], size=[-1,1]), [-1]), tf.int32), 20)		
		# _c6_onehot = tf.one_hot(tf.cast(tf.reshape(tf.slice(_column, begin=[0, 6], size=[-1,1]), [-1]), tf.int32), 20)		
		# _c7_onehot = tf.one_hot(tf.cast(tf.reshape(tf.slice(_column, begin=[0, 7], size=[-1,1]), [-1]), tf.int32), 20)		
		# _c8_onehot = tf.one_hot(tf.cast(tf.reshape(tf.slice(_column, begin=[0, 8], size=[-1,1]), [-1]), tf.int32), 20)		
		# _c9_onehot = tf.one_hot(tf.cast(tf.reshape(tf.slice(_column, begin=[0, 9], size=[-1,1]), [-1]), tf.int32), 20)		

		# 全连接层输入（现在考虑把每一行的高度也转成onehot向量输入，可能会更符合神经元的输入，同时增加隐藏层神经元的数量）
		# ——下次重新训练时可以考虑上述改动，同时把评价函数的失败惩罚调整为-100
		# 上面的完成了，下一次的修改，打算把池化层去掉，因为我的模型需要精确计算，而之前的效果里经常有对错位置的情况
		h_fc_input_rx = tf.concat([h_pool3_flat, _block_onehot, _nextblock_onehot \
			# , _c0_onehot \
			# , _c1_onehot \
			# , _c2_onehot \
			# , _c3_onehot \
			# , _c4_onehot \
			# , _c5_onehot \
			# , _c6_onehot \
			# , _c7_onehot \
			# , _c8_onehot \
			# , _c9_onehot \
			], 1)
		h_fc_input_rx_len = h_pool3_flat_len + 7 + 7 #+ 10 * 20

		# 全连接层
		fc_layer1_size = 1024
		# fc_layer2_size = 512
		# fc_layer3_size = 512
		W_fc1_rx = weight_variable([h_fc_input_rx_len, fc_layer1_size])
		b_fc1_rx = bias_variable([fc_layer1_size])
		h_fc1_rx = tf.nn.relu6(tf.matmul(h_fc_input_rx, W_fc1_rx) + b_fc1_rx)
		print("W_fc1_rx", W_fc1_rx)
		print("h_fc1_rx", h_fc1_rx)

		#layer fc2 for rx
		# 可以考虑在第二个隐藏车再输入一次当前方块的索引，下一次尝试
		# W_fc2_rx = weight_variable([fc_layer1_size + 7,fc_layer2_size])
		# b_fc2_rx = bias_variable([fc_layer2_size])
		# h_fc2_rx = tf.nn.relu6(tf.matmul(tf.concat([h_fc1_rx, _block_onehot], 1), W_fc2_rx) + b_fc2_rx)
		# print("W_fc2_rx", W_fc2_rx)
		# print("h_fc2_rx", h_fc2_rx)

		#layer fc3 for rx
		# W_fc3_rx = weight_variable([fc_layer2_size + 7,fc_layer3_size])
		# b_fc3_rx = bias_variable([fc_layer3_size])
		# h_fc3_rx = tf.nn.relu6(tf.matmul(tf.concat([h_fc2_rx, _block_onehot], 1), W_fc3_rx) + b_fc3_rx)
		# print("W_fc3_rx", W_fc3_rx)
		# print("h_fc3_rx", h_fc3_rx)

		#模型输出
		W_rx = weight_variable([fc_layer1_size, 40])
		b_rx = bias_variable([40], init=0.0)
		output = tf.add(tf.matmul(h_fc1_rx, W_rx), b_rx, name="output")
		print("output", output)

	return model
