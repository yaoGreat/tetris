import tensorflow as tf
from game import Tetris

model = None
sess = None

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def create_model():
	model = tf.Graph()
	with model.as_default():
		#input
		_tiles = tf.placeholder(tf.float32, [None, 20, 10], name="tiles")
		_current = tf.placeholder(tf.float32, [None, 2], name="current") # idx, next_idx
		_action = tf.placeholder(tf.int32, [None, 2], name="action") # x, rotate
		keep_prob = tf.placeholder(tf.float32, name="kp")
		print("_tiles", _tiles)
		print("_current", _current)
		print("_action", _action)

		#layer 1
		_tiles_reshape = tf.reshape(_tiles, [-1, 20, 10, 1])
		W_conv1 = weight_variable([5,5,1,32])
		b_conv1 = bias_variable([32])
		h_conv1 = tf.nn.relu(conv2d(_tiles_reshape, W_conv1) + b_conv1)
		h_pool1 = max_pool_2x2(h_conv1)
		print("h_pool1", h_pool1)

		#layer fc1
		W_fc1 = weight_variable([10 * 5 * 32 + 2, 1024])
		b_fc1 = bias_variable([1024])
		h_pool_flat = tf.reshape(h_pool1, [-1, 10 * 5 * 32])
		h_fc1_input = tf.concat([h_pool_flat, _current], 1)
		h_fc1 = tf.nn.relu(tf.matmul(h_fc1_input, W_fc1) + b_fc1)
		print("h_fc1_input", h_fc1_input)
		print("h_fc1", h_fc1)

		#drop out
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
		print("h_fc1_drop", h_fc1_drop)

		#layer fc2 x * 4 + r
		W_fc2_xr = weight_variable([1024, 40])
		b_fc2_xr = bias_variable([40])
		xr_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2_xr) + b_fc2_xr)
		print("xr_conv", xr_conv)

		#onehot of input train _action x * 4 + r
		x_, r_ = tf.split(_action, 2, 1)
		xr = tf.reshape(x_, [-1]) * 4 + tf.reshape(r_, [-1])
		hot_xr = tf.one_hot(xr, 40, 1.0, 0.0, -1, name="one_hot_xr")
		print("hot_xr", hot_xr)

		#train
		cross_entropy = -tf.reduce_sum(hot_xr * tf.log(xr_conv))
		train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, name="train")
		print("train_step", train_step)

		#output
		output = tf.argmax(xr_conv, 1, name="output")
		print("output", output)

	return model

def init_train():
	global model
	global sess
	model = create_model()
	sess = tf.InteractiveSession(graph = model)
	cp = tf.train.latest_checkpoint('model_0/')
	if cp == None:
		print("init model with default val")
		tf.global_variables_initializer().run()
	else:
		print("init model with saved val")
		saver = tf.train.Saver(max_to_keep = 1)
		saver.restore(sess, cp)

def do_train(tetris):
	global model
	global sess
	tiles = [tetris.tiles()]
	current = [[tetris.current_index(), tetris.next_index()]]
	action = [[tetris.current_X(), tetris.current_rotate()]]
	kp = 0.5
	train = model.get_operation_by_name("train")
	sess.run(train, feed_dict={"tiles:0":tiles, "current:0":current, "action:0":action, "kp:0":kp})

def save_train():
	global sess
	saver = tf.train.Saver(max_to_keep = 1)
	saver.save(sess, 'model_0/save.ckpt')

__cur_step = -1
__cur_output = 0
def run_game(tetris):
	global model
	global sess
	global __cur_step
	global __cur_output
	if tetris.step() != __cur_step:
		tiles = [tetris.tiles()]
		current = [[tetris.current_index(), tetris.next_index()]]
		kp = 1
		output = model.get_tensor_by_name("output:0")
		__cur_output = output.eval(feed_dict={"tiles:0":tiles, "current:0":current, "kp:0":kp})
		__cur_step = tetris.step()
		print("step %d, output: %d, x: %d, r: %d" % (__cur_step, __cur_output, int(__cur_output / 4), int(__cur_output % 4)))
	
	x = int(__cur_output / 4)
	r = int(__cur_output % 4)
	if tetris.move_step_by_ai(x, r):
		tetris.fast_finish()

if __name__ == '__main__':
	init_train()
	finish_train()
