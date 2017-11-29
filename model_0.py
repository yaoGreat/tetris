import tensorflow as tf
import numpy as np
from collections import deque
import random
import copy
from game import Tetris
from play import TetrisUI

model = None
sess = None
saver = None

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
		_action = tf.placeholder(tf.float32, [None, 4 * 10], name="action") # x * 4 + rotate
		_target = tf.placeholder(tf.float32, [None], name="target") # reward + gamma * max(Q_sa)
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
		xr_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2_xr) + b_fc2_xr, name="argmax_xr")
		print("xr_conv", xr_conv)

		#output
		output = tf.argmax(xr_conv, 1, name="output")
		print("output", output)

		#train
		action = tf.reduce_sum(tf.multiply(xr_conv, _action), reduction_indices = 1)
		cost = tf.reduce_mean(tf.square(action - _target), name="cost")
		optimizer = tf.train.AdamOptimizer(1e-6).minimize(cost, name="train_op")
		print("optimizer", optimizer)

	return model

def init_model():
	global model
	global sess
	global saver
	model = create_model()
	sess = tf.InteractiveSession(graph = model)
	saver = tf.train.Saver(max_to_keep = 1)
	cp = tf.train.latest_checkpoint('model_0/')
	if cp == None:
		print("init model with default val")
		tf.global_variables_initializer().run()
	else:
		print("init model with saved val")
		saver.restore(sess, cp)

def save_model():
	global sess
	global saver
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

def train(tetris
	, memory_size = 1000
	, batch_size = 50
	, train_steps = 10000
	, gamma = 0.95
	, init_epsilon = 1
	, min_epsilon = 0.01
	, ui = None):
	global model
	global sess
	D = deque()

	epsilon = init_epsilon
	step = 0
	last_cost = 0
	status_0 = train_make_status(tetris)
	while True:
		#run game
		action_0 = None # judge action, random or from model
		if random.random() < epsilon:
			action_0 = [0] * 40
			action_0[random.randrange(40)] = 1
		else:
			action_0 = train_cal_action([status_0], model)[0]
		epsilon = init_epsilon + (min_epsilon - init_epsilon) * step / train_steps

		reward_1 = train_run_game(tetris, action_0, ui)  #use the action to run, then get reward
		gameover_1 = 0
		if tetris.gameover():
			tetris.reset()
			gameover_1 = 1
		status_1 = train_make_status(tetris)

		#log to memory
		D.append((status_0, action_0, reward_1, status_1, gameover_1))
		if len(D) > memory_size:
			D.popleft()

		#review memory
		if step > batch_size:
			batch = random.sample(D, batch_size)
			status_0_batch = [d[0] for d in batch]
			action_0_batch = [d[1] for d in batch]
			reward_1_batch = [d[2] for d in batch]
			status_1_batch = [d[3] for d in batch]
			gameover_1_batch = [d[4] for d in batch]

			action_1_batch = train_cal_action(status_1_batch, model)
			target_batch = []
			for i in range(len(batch)):
				if gameover_1_batch[i] == 1:
					target_batch.append(reward_1_batch[i])
				else:
					target_batch.append(reward_1_batch[i] + gamma * np.max(action_1_batch[i]))

			last_cost = train_run_train_op(status_0_batch, action_0_batch, target_batch, model)

		#loop
		status_0 = status_1
		step += 1
		if step > train_steps:
			break

		if step % 100 == 1:
			info = "train step %d, epsilon: %f, cost: %f" % (step, epsilon, last_cost)
			if ui == None:
				print(info)
			else:
				ui.show_train_info(info)
			save_model()

def train_make_status(tetris):	# 0, tiles; 1, current
	tiles = copy.deepcopy(tetris.tiles())
	current = [tetris.current_index(), tetris.next_index()]
	status = {"tiles":tiles, "current":current}
	return status

def train_cal_action(status_s, use_model):
	global sess
	tiles = [status["tiles"] for status in status_s]
	current = [status["current"] for status in status_s]
	kp = 1
	argmax_xr = use_model.get_tensor_by_name("argmax_xr:0").eval(feed_dict={"tiles:0":tiles, "current:0":current, "kp:0":kp})
	argmax = [np.argmax(x) for x in argmax_xr]
	action = np.eye(40)[argmax]
	return action

def train_run_train_op(status_s, action_s, target_s, use_model):
	global sess
	tiles = [status["tiles"] for status in status_s]
	current = [status["current"] for status in status_s]
	kp = 1
	train_op = model.get_operation_by_name("train_op")
	cost = model.get_tensor_by_name("cost:0")
	_, cost_val = sess.run((train_op, cost), feed_dict={"tiles:0":tiles, "current:0":current, "kp:0":kp, "action:0":action_s, "target:0":target_s})
	return cost_val

def train_run_game(tetris, action, ui):
	xr = np.argmax(action)
	x = int(xr / 4)
	r = int(xr % 4)

	old_scores = train_cal_scores(tetris)

	while True:
		r = tetris.move_step_by_ai(x, r)

		if ui != None:
			if ui.refresh_and_check_quit():
				raise Exception("user quit")

		if r:
			tetris.fast_finish()
			break

	new_scores = train_cal_scores(tetris)
	reward = train_cal_reward(old_scores, new_scores)
	return reward

def train_cal_scores(tetris):
	row_cnt = 0
	total_fill = 0

	tiles = tetris.tiles()
	for row in tiles:
		row_fill = 0
		for t in row:
			if t > 0:
				row_fill += 1
		if row_fill > 0:
			row_cnt += 1
			total_fill += row_fill

	fill_rate = 0
	if row_cnt > 0:
		fill_rate = float(total_fill) / float(row_cnt * tetris.width())

	gameover = 0
	if tetris.gameover():
		gameover = 1
	return {"score":tetris.score(), "row_cnt":row_cnt, "fill_rate":fill_rate, "gameover":gameover}

reward_weight_score = 1
reward_weight_row_inc = -0.5
reward_weight_fill_rate = 1
normal_row_inc = 1
def train_cal_reward(old_scores, new_scores):
	if new_scores["gameover"] == 1:
		return -1

	score_inc = new_scores["score"] - old_scores["score"]
	row_inc = new_scores["row_cnt"] - old_scores["row_cnt"]
	fill_inc = new_scores["fill_rate"] - old_scores["fill_rate"]

	if score_inc > 0:
		return 1

	if row_inc < 1:
		return 1
	elif row_inc == 1:
		return 0
	else:
		return -1
	#return score_inc * reward_weight_score + (row_inc - normal_row_inc) * reward_weight_row_inc + fill_inc * reward_weight_fill_rate

if __name__ == '__main__':
	init_model()
	save_model()
