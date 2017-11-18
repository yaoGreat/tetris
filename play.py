import curses
from game import Tetris

key_up = 65
key_down = 66
key_left = 68
key_right = 67
key_space = 32

class TetrisUI:
	__baseX = 4
	__baseY = 2
	__tileWidth = 2
	__tetris = None
	__scr = None
	__lastkey = 0

	def __init__(self, tetris):
		print("init tetris gui")
		self.__tetris = tetris
		self.__scr = curses.initscr()
		curses.noecho()
		curses.cbreak()
		self.__scr.timeout(500)

	def __del__(self):
		self.__tetris = None
		self.__scr = None
		curses.nocbreak()
		curses.echo()
		curses.endwin()
		print("destory tetris gui")

	def __refresh(self):
		self.__scr.clear()
		self.__scr.border()
		tiles = self.__tetris.tiles()
		width = self.__tetris.width()
		height = self.__tetris.height()

		for y in range(0, height):
			for x in range(0, width):
				tile = tiles[y][x]
				self.__drawtile(x, y, tile)

		current = self.__tetris.current()
		for t in current:
			self.__drawtile(t[0], t[1], t[2])

		info_x = (self.__tetris.width() + 3) * self.__tileWidth
		info_y = self.__baseY + 8

		_next = self.__tetris.next()
		for t in _next:
			self.__drawtile(t[0], t[1], t[2], baseX = info_x)

		self.__drawcontent(info_x, info_y, "INFO")
		self.__drawcontent(info_x, info_y + 1, "Score: %d" % self.__tetris.score())
		self.__drawcontent(info_x, info_y + 2, "Step: %d" % self.__tetris.step())
		self.__drawcontent(info_x, info_y + 3, "LastKey: %d" % self.__lastkey)
		self.__drawcontent(info_x, info_y + 4, "Dbg: %s" % self.__tetris.dbginfo())
		if self.__tetris.gameover():
			self.__drawcontent(info_x, info_y + 5, "GAME OVER")


	def __drawtile(self, x, y, v, baseX = 0, baseY = 0):
		ch = '.'
		if v != 0:
			ch = str(v)
		if baseX == 0:
			baseX = self.__baseX
		if baseY == 0:
			baseY = self.__baseY
		self.__drawcontent(baseX + x * self.__tileWidth, baseY + y, ch)

	def __drawcontent(self, x, y, s):
		self.__scr.addstr(y, x, s)

	def loop(self, ai_model = None):
		while True:
			self.__refresh()
			self.__tetris.clear_dbginfo()
			c = self.__scr.getch()

			if self.__tetris.gameover():
				if c == ord('q'):
					print("exit")
					break
				else:
					continue

			if c < 0:
				if ai_model == None:
					self.__tetris.move_current(y = 1)
				else:
					ai_model.run_game(self.__tetris)
			elif c == key_left:
				if ai_model == None:
					self.__tetris.move_current(x = -1)
			elif c == key_right:
				if ai_model == None:
					self.__tetris.move_current(x = 1)
			elif c == key_down:
				if ai_model == None:
					self.__tetris.move_current(y = 1)
			elif c == key_up:
				if ai_model == None:
					self.__tetris.rotate_current()
			elif c == key_space:
				if ai_model == None:
					self.__tetris.fast_finish()
			elif c == ord('q'):
				print("exit")
				break
			
			if c > 0:
				self.__lastkey = c
			
import model_0 as model

def play_train():
	game = Tetris()
	model.init_train()
	game.set_train_mode(model)
	ui = TetrisUI(game)
	ui.loop()
	del ui
	model.save_train()
	del game

def play_ai():
	game = Tetris()
	model.init_train()
	ui = TetrisUI(game)
	err = None
	try:
		ui.loop(ai_model = model)
	except Exception:
		err = "Exception"
	del ui
	del game
	if err != None:
		print(err)

def play_ai_without_ui():
	game = Tetris()
	model.init_train()
	while not game.gameover():
		model.run_game(game)
	del game

if __name__ == '__main__':
	play_train()
	# play_ai()
	# play_ai_without_ui()
