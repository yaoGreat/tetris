﻿import tensorflow as tf
import numpy as np
from collections import deque
from time import sleep
import time
import datetime
import random
import copy
from game import Tetris
from play import TetrisUI
import model_0 as using_model
import mcts

# TODO: 
# 1、完善模型，有几个思路：增加卷积层，为每个方块类型设置单独的矩阵参数，分离位置和旋转两个操作、但是还没有想好
# 2、将模型分为训练模型和目标模型（target），target用于在回顾记忆时，计算n+i的maxQ。尝试修改了两次，都没成功，需要研究一下sess和saver之类的接口。目前来看，这一条的影响似乎不是最大的。
# 3、奖励函数的算法，增加更详细的算法，更精确的评估操作

# 发现了一个问题，模型计算出来的权值，都nan了，貌似是卷积加全连接之后，结果太大。怎么解决还需要考虑，是变量初始值的问题，还是什么
# 也许需要通过输出，找到再网络的哪一层出现nan。现在看来，距离调整算法还有段距离呢，先把模型的bug调好

# 经过调试，发现随着训练步骤增加，Q值会越来越大。貌似，某一个action的Q值随着训练变大之后，后续所有计算的Q值都会随着增大，这导致了一个正反馈放大效应，结果就是inf
# 貌似应该修改一下模型，增加一个层吧。另外，两个sess是否应该加一下了？现在每次训练之后都会更新target，也许会导致正反馈效应

# 模型已经加层了，一个卷积层，一个全连接层，但是数值还是会inf，所有，考虑一下多个sess吧

# 收敛的问题已经解决了，现在有了一个最简单的，能够玩一种类型方块的ai
# 有一个没有理解的问题是：之前的网路输入带有游戏中的随机值，导致算法的不确定性，但是不知为何，去掉这个不确定性，算法的效果反而差一些。还没理解。
# 更新了一个新的模型，把4个旋转方向的网络独立运算。下一步，可以把这个算法重构一下，方便针对所有方向进行统一的调整和优化。现在的网络速度有点慢。
# 接下来，可以做两件事：调整奖励函数；实验另一个带有旋转的方块看看。

# 现在的模型，对于顶端的状态貌似没有足够的重视
# model_2 训练4W次，可以解决T型的游戏了，下一步，多个方块

# 先用现有模型测试一下，如果不行，现在有这个思路：
# 减少一下dnn的层数，然后为当前方块的index建立一个onehot向量，然后建立7组dnn网络，每种对应一个方块形状，这样，会有7*4=28个dnn网络，为每种情况进行判断
# 测试结束，原来的网络——失败

model = None
sess = None
saver = None
is_new_model = False
save_path = ""

def init_model(train = False, forceinit = False, learning_rate = 0):
	global model
	global sess
	global saver
	global is_new_model
	global save_path
	
	# 初始化模型和路径
	model = using_model.create_model_4()
	save_path = "model_4/"
	
	with model.as_default():
		global_step = tf.Variable(0, name="step")

	if train:
		create_train_op(model, learning_rate=learning_rate)
	
	sess_config=tf.ConfigProto(device_count={"CPU":8}, inter_op_parallelism_threads=0, intra_op_parallelism_threads=0)
	sess = tf.Session(graph = model, config=sess_config) #这里使用普通的Session，是为了使用多个sess做准备

	with model.as_default():
		saver = tf.train.Saver(max_to_keep = 1)

		cp = tf.train.latest_checkpoint(save_path)
		if cp == None or forceinit:
			print("init model with default val")
			tf.global_variables_initializer().run(session=sess)
			save_model()
			is_new_model = True
		else:
			print("init model with saved val")
			saver.restore(sess, cp)
			is_new_model = False

def save_model():
	global sess
	global saver
	global save_path
	saver.save(sess, save_path + 'save.ckpt')

def restore_model(dst_sess):
	global saver
	global save_path
	cp = tf.train.latest_checkpoint(save_path)
	if cp != None:
		saver.restore(dst_sess, cp)
	else:
		print("restore model fail.")

__cur_step = -1
__cur_output = 0
__use_mcts = False
def run_game(tetris):
	global model
	global sess
	global __cur_step
	global __cur_output
	if tetris.step() != __cur_step:
		status = train_make_status(tetris)
		__cur_step = tetris.step()
		if __use_mcts:
			__cur_output = mcts.mcts_search(status, _status_func=train_make_status
				, _reward_func = train_cal_reward
				, _weight_func = lambda s : train_cal_action_weight([s], model, sess)[0]
				, _run_func = train_run_game)
		else:
			__cur_output_xr = train_cal_action_weight([status], model, sess)[0]
			__cur_output = np.argmax(__cur_output_xr)
		print("step %d, score: %d, output: %d, x: %d, r: %d" % (__cur_step, tetris.score(), __cur_output, int(__cur_output / 4), int(__cur_output % 4)))

	x, r = train_getxr_by_action(__cur_output)
	if tetris.move_step_by_ai(x, r):
		tetris.fast_finish()


def create_train_op(model, learning_rate):
	with model.as_default():
		#train input
		_action = tf.placeholder(tf.float32, [None, 40], name="action")
		_targetQ = tf.placeholder(tf.float32, [None], name="targetQ") # reward + gamma * max(Q_sa)

		#train
		output = model.get_tensor_by_name("output:0")

		# 个人感觉用max好一点，因为这样可以让tf知道，只有最大的值是有意义的。而用sum，则所有的分量都会参与运算（虽然传入的其他分量都是0）
		Q = tf.reduce_sum(tf.multiply(output, _action), reduction_indices = 1, name="Q")	# take the weight of _action in output as Q
		# Q = tf.reduce_max(tf.multiply(output, _action), reduction_indices = 1, name="Q")	# take the weight of _action in output as Q
		
		cost = tf.reduce_mean(tf.square(Q - _targetQ), name="cost")
		
		# 用梯度下降，则数值会变的越来越大，还不知道是什么原因
		# optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cost, name="train_op")
		init_lr = 1e-4
		if learning_rate != 0:
			init_lr = learning_rate

		# 0.98 ^ 100 = 0.13，所以500表示每5W次训练，学习率降低1个数量级
		global_step = model.get_tensor_by_name("step:0")
		decay_lr = tf.train.exponential_decay(init_lr, global_step, decay_steps=400, decay_rate=0.98, staircase=True)
		optimizer = tf.train.AdamOptimizer(init_lr).minimize(cost, name="train_op", global_step=global_step)
		print("optimizer", optimizer)
		print("init learning rate is: %f" % init_lr)

	return model

def train(tetris
	, memory_size = 1000
	, batch_size = 50
	, train_steps = 10000
	, gamma = 0.6
	, init_epsilon = 1
	, min_epsilon = 0.01
	, savePerStep = 100
	, ui = None):
	global model
	global sess
	global is_new_model
	D = deque()

	target_sess = tf.Session(graph = model)
	restore_model(target_sess)

	if not is_new_model:
		init_epsilon = float(init_epsilon) / 2

	epsilon = init_epsilon
	step = 0
	status_0 = train_make_status(tetris)
	print("train start at: " + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
	while True:
		#run game
		action_0 = [0] * 40 # judge action, random or from model, this action vector must be onehot
		if random.random() < epsilon:
			action_0[random.randrange(40)] = 1
		else:
			idx = np.argmax(train_cal_action_weight([status_0], model, sess)[0])
			action_0[idx] = 1
		epsilon = init_epsilon + (min_epsilon - init_epsilon) * step / train_steps

		gameover = train_run_game(tetris, action_0, ui)  #use the action to run, then get reward
		status_1 = train_make_status(tetris)
		reward_1, reward_info = train_cal_reward(tetris, gameover)
		
		#log to memory
		D.append((status_0, action_0, reward_1, status_1, gameover))
		if len(D) > memory_size:
			D.popleft()

		if ui != None:
			weight = train_cal_action_weight([status_0], model, sess)[0]
			ui.log("action: %d, maxweight: %f, reward: %f, info: %s" % (np.argmax(action_0), np.max(weight), reward_1, reward_info))

		#review memory
		if len(D) > batch_size:
			batch = random.sample(D, batch_size)
			status_0_batch = [d[0] for d in batch]
			action_0_batch = [d[1] for d in batch]
			reward_1_batch = [d[2] for d in batch]
			status_1_batch = [d[3] for d in batch]
			gameover_1_batch = [d[4] for d in batch]

			t_calnet_begin = datetime.datetime.now()
			Q_1_batch = train_cal_action_weight(status_1_batch, model, target_sess)	#action_1 == Q_i+1
			t_calnet_use = datetime.datetime.now() - t_calnet_begin

			targetQ_batch = []
			for i in range(len(batch)):
				if gameover_1_batch[i]:
					targetQ_batch.append(reward_1_batch[i])
				else:
					# 不同形状本身的价值不同，这里是否需要考虑一下？还没想好
					targetQ_batch.append(reward_1_batch[i] + gamma * np.max(Q_1_batch[i]))

			tiles = [status["tiles"] for status in status_0_batch]
			column = [status["column"] for status in status_0_batch]
			current = [status["current"] for status in status_0_batch]
			kp = 1
			train_op = model.get_operation_by_name("train_op")

			t_trainnet_begin = datetime.datetime.now()
			_, _output, _Q, _cost, _step = sess.run((train_op
				, model.get_tensor_by_name("output:0")
				, model.get_tensor_by_name("Q:0")
				, model.get_tensor_by_name("cost:0")
				, model.get_tensor_by_name("step:0")
				# , model.get_tensor_by_name("W_conv1:0")
				# , model.get_tensor_by_name("b_conv1:0")
				)
				, feed_dict={"tiles:0":tiles, "column:0":column, "current:0":current, "action:0":action_0_batch, "targetQ:0":targetQ_batch, "kp:0":kp})
			t_trainnet_use = datetime.datetime.now() - t_trainnet_begin

			if step % savePerStep == 0:
				match_cnt = 0
				for i in range(batch_size):
					if targetQ_batch[i] != 0 and float(abs(_Q[i] - targetQ_batch[i])) / float(abs(targetQ_batch[i])) < 0.1:
						match_cnt += 1
				match_rate = float(match_cnt) / float(batch_size)
				info = "train step %d(g: %d), epsilon: %f, action[0]: %d, reward[0]: %f, targetQ[0]: %f, Q[0]: %f, matchs: %f, cost: %f (time: %d/%d)" \
						% (step, _step, epsilon, np.argmax(action_0_batch[0]), reward_1_batch[0], targetQ_batch[0], _Q[0], match_rate, _cost \
						, t_calnet_use.microseconds, t_trainnet_use.microseconds)
				if ui == None:
					print(info)
					# print("W1: ", _W1)
					# print("b1: ", _b1)
					if savePerStep == 1:	#为了调试，这样能看清楚日志
						sleep(1)
				else:
					ui.log(info)
				save_model()
				restore_model(target_sess)

		#loop
		status_0 = status_1
		step += 1
		if step > train_steps:
			break

	print("train finish at: " + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
		

def train_make_status(tetris):	# 0, tiles; 1, current
	w = tetris.width()
	h = tetris.height()
	image = [[ 0 for x in range(w) ] for y in range(h)]
	column_height = [0] * 10
	
	tiles = tetris.tiles()
	for y in range(0, h):
		for x in range(0, w):
			if tiles[y][x] > 0:
				# image[y][x] = tiles[y][x]
				image[y][x] = 1
				column_height[x] = max(column_height[x], 20 - y)

	# current = tetris.current()
	# for t in current:
	# 	x = t[0]
	# 	y = t[1]
	# 	image[y][x] = 1
	
	cur_block_idx = [tetris.current_index(), tetris.next_index()]
	score = tetris.score()
	step = tetris.step()
	status = {"tiles":image, "current":cur_block_idx, "column":column_height, "score":score, "step":step}
	return status

def train_cal_action_weight(status_s, use_model, use_sess):
	tiles = [status["tiles"] for status in status_s]
	current = [status["current"] for status in status_s]
	column_height = [status["column"] for status in status_s]
	kp = 1
	argmax_xr = use_sess.run(use_model.get_tensor_by_name("output:0"), feed_dict={"tiles:0":tiles, "column:0":column_height, "current:0":current, "kp:0":kp})
	return argmax_xr

def train_getxr_by_action(action):
	r = int(action / 10)
	x = int(action % 10)
	return x, r

def train_run_game(tetris, action, ui):
	xr = np.argmax(action)
	# x = int(xr / 4)
	# r = int(xr % 4)
	x, r = train_getxr_by_action(xr)

	while True:
		move_finish = tetris.move_step_by_ai(x, r)

		if ui != None:
			if ui.refresh_and_check_quit():
				raise Exception("user quit")

		if move_finish:
			tetris.fast_finish()
			break

	gameover = False
	if tetris.gameover():
		tetris.reset()
		gameover = True

	return gameover


s_column_height = None
s_column_hole = None
s_fill_rate = 0

def train_reset_reward_status():
	global s_column_height
	global s_column_hole
	global s_fill_rate
	s_column_height = [0] * 10
	s_column_hole = [0] * 10
	s_fill_rate = 0

def train_cal_reward(tetris, gameover = False):
	# 希望统计的内容：
	# 行数的增量，被遮挡的空格数量，填充率
	# 还可以增加一个高度差的属性
	global s_column_height
	global s_column_hole
	global s_fill_rate

	if gameover:
		train_reset_reward_status()
		return -100, ""

	if s_column_height == None:
		train_reset_reward_status()

	row_cnt = 0
	total_fill = 0
	top_y_index = [20] * 10 # 临时数组，计算每一列中最顶端的y值，最上方为0，最下面是19
	column_hole = [0] * 10

	tiles = tetris.tiles()
	for y in range(len(tiles)):
		row = tiles[y]
		row_fill = 0
		for x in range(len(row)):
			t = row[x]
			if t > 0:
				row_fill += 1
				top_y_index[x] = min(top_y_index[x], y)
			else:
				if y > top_y_index[x]:
					column_hole[x] += 1
		if row_fill > 0:
			row_cnt += 1
			total_fill += row_fill


	column_height = [20 - y for y in top_y_index]
	fill_rate = float(total_fill) / float(row_cnt * len(tiles[0])) if row_cnt > 0 else 0

	erase_row = tetris.last_erase_row()
	inc_row = max(column_height) - max(s_column_height)
	inc_hole = sum(column_hole) - sum(s_column_hole)
	inc_fill_rate = fill_rate - s_fill_rate
	inc_var = np.array(column_height).var() - np.array(s_column_height).var() #每一列高度的方差，表示最顶层平整的程度

	info = "erase_row: %d, inc_fill: %f, inc_row: %d, inc_hole: %d, inc_var: %f" % (erase_row, inc_fill_rate, inc_row, inc_hole, inc_var)
	# reward = (float(erase_row) * 10 + float(inc_fill_rate) * 100 - float(inc_row) * pow(1.05, max(column_height)) - float(inc_hole) * 2 - float(inc_var)) #/ 100.0
	reward = (float(erase_row) * 15 - float(inc_row) * pow(1.1, max(column_height)) - float(inc_hole) * 4 - float(inc_var) * 2)

	s_column_height = column_height
	s_column_hole = column_hole
	s_fill_rate = fill_rate
	return reward, info

if __name__ == '__main__':
	init_model()
	save_model()
