from game import Tetris
from datetime import datetime
import numpy as np
import math

print_info = False
pick_action_count = 5
mcts_c = 1.96
search_max_count = 100
search_max_time = 500000 #微秒

status_func = None
reward_func = None
weight_func = None
run_func = None

g_node_id = 0
class mcts_node:
	__id = 0
	__status = None
	__action = 0
	__children = None
	__parent = None
	__visit = 0
	__Q = 0
	__over = False
	__valid_action = None

	def __init__(self, _status, _action, _initQ, _over, _parent):
		global g_node_id
		g_node_id += 1
		self.__id = g_node_id
		self.__status = _status
		self.__action = _action
		self.__children = {}
		self.__parent = _parent
		self.__visit = 1
		self.__Q = _initQ
		self.__over = _over
		self.__valid_action = []

		action_weigth = weight_func(_status)
		for i in range(pick_action_count):
			action = np.argmax(action_weigth)
			action_weigth[action] = -10000
			self.__valid_action.append(action)

		# for i in range(40):
		# 	self.__valid_action.append(i)

	def get_id(self):
		return self.__id

	def get_status(self):
		return self.__status

	def valid_action(self):
		return self.__valid_action

	def pop_valid_action(self):
		if len(self.__valid_action) > 0:
			return self.__valid_action.pop(0)
		else:
			return -1

	def add_child(self, action, child):
		self.__children[action] = child

	def addQ(self, q):
		self.__Q += q

	def addVisit(self, v):
		self.__visit += v

	def getParent(self):
		return self.__parent

	def getChildren(self):
		return self.__children

	def getBestChild(self, c):
		maxK = 0
		maxChild = None
		for action in self.__children:
			child = self.__children[action]
			t = 2.0 * math.log(float(self.getVisit())) / float(child.getVisit())
			k = float(child.getQ()) / float(child.getVisit()) + c * math.sqrt(t)
			if k > maxK:
				maxK = k
				maxChild = child
		if maxChild == None:
			print("getBestChild error, child cnt: %d" % len(self.__children))
			print(self.__children)
		return maxChild

	def getQ(self):
		return self.__Q

	def getVisit(self):
		return self.__visit

	def isOver(self):
		return self.__over

	def getAction(self):
		return self.__action

def mcts_search(status0, _status_func, _reward_func, _weight_func, _run_func):
	global status_func
	global reward_func
	global weight_func
	global run_func
	status_func = _status_func
	reward_func = _reward_func
	weight_func = _weight_func
	run_func = _run_func
	
	n0 = mcts_node(status0, _action = 0, _initQ = 0, _over = False, _parent = None)

	t_begin = datetime.now()

	while (datetime.now() - t_begin).microseconds < search_max_time:
		node, action = mcts_get_action_to_try(n0)
		if node == None:
			break
		new_node = mcts_do_action(node, action)
		mcts_backpropagation(new_node)

	if print_info:
		mcts_dump_tree2(n0, 0)
	bestChild = n0.getBestChild(0)
	return bestChild.getAction()

def mcts_get_action_to_try(n0):
	node = n0
	while not node.isOver():
		if len(node.valid_action()) > 0:
			action = node.pop_valid_action()
			return node, action
		else:
			node = node.getBestChild(mcts_c)
	return None, -1

tetris_game = None
def mcts_do_action(n, a):	
	global tetris_game
	if tetris_game == None:
		tetris_game = Tetris()

	state = n.get_status()
	nodes = state["tiles"]
	current = state["current"]
	score = state["score"]
	step = state["step"]
	tetris_game.apply_status_by_ai(nodes = nodes, _current=current[0], _next=current[1], _score=score, _step=step)
	
	reward_func(tetris_game, False)
	onehot_action = [0] * 40
	onehot_action[a] = 1
	gameover = run_func(tetris_game, onehot_action, None)
	reward, info = reward_func(tetris_game, gameover)

	q = math.exp(reward)
	s1 = status_func(tetris_game)
	# q, info = mcts_calulate_status_q(tetris_game)
	n1 = mcts_node(s1, _action = a, _initQ = q, _over = gameover, _parent = n)
	n.add_child(a, n1)
	# if print_info:
	# 	print("add child %d -> %d(%d/%f) -> %d, i: %s" % (n.get_id(), current[0], a, n1.getQ(), n1.get_id(), info))
	return n1

def mcts_backpropagation(n):
	node = n.getParent()
	q = n.getQ()
	while node != None:
		node.addVisit(1)
		node.addQ(q)
		node = node.getParent()

def mcts_calulate_status_q(tetris):
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

	hv = np.array(column_height).var()
	q = (20-row_cnt) * 2 + fill_rate * 100 - hv * 2 + tetris.score() - sum(column_hole)
	info = "q: row: %d, fill: %f, chv: %f, score: %d, hole: %d" % (row_cnt, fill_rate, hv, tetris.score(), sum(column_hole))
	return max(q, 0), info

def mcts_dump_tree(n0):
	q = []
	q.append(n0)
	while(len(q) > 0):
		n = q.pop(0)
		print("node: %d, parent: %d, action: %d, q: %f, visit: %d" % (n.get_id(), n.getParent().get_id() if n.getParent() != None else 0 \
			, n.getAction(), n.getQ(), n.getVisit()))
		children = n.getChildren()
		for a in children:
			child = children[a]
			q.append(child)

def mcts_dump_tree2(n, lv):
	info = ""
	for i in range(lv):
		info += "    "
	info += "id: %d, action: %d, q: %f, visit: %d" % (n.get_id(), n.getAction(), n.getQ(), n.getVisit())
	print(info)
	children = n.getChildren()
	for a in children:
		child = children[a]
		mcts_dump_tree2(child, lv + 1)
