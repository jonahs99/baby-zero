""" This is a connect4 implementation using bitboards
"""

import array
import bitboard as bb

bb_str = bb.make_rep(6, 7)

line_verti = bb.from_pos([0, 7, 14, 21])
line_horiz = bb.from_pos([0, 1, 2, 3])
line_maind = bb.from_pos([0, 8, 16, 24])
line_antid = bb.from_pos([21, 15, 9, 3])

lines = [ line_verti << (c + r*7) for c in range(7) for r in range(3) ] + \
		[ line_horiz << (c + r*7) for c in range(4) for r in range(6) ] + \
		[ line_maind << (c + r*7) for c in range(4) for r in range(3) ] + \
		[ line_antid << (c + r*7) for c in range(4) for r in range(3) ]
lines_including = [ [l for l in lines if bb.get_bit(l, pos)] for pos in range(42) ]
columns = [ bb.from_pos([r*7+i for r in range(6)]) for i in range(7) ]
full = bb.from_pos(range(42))

class State:
	"""
	Bitboards are arranged like
	35 36 37 38 39 40 41
	...
	7  8  9  10 11 12 13
	0  1  2  3  4  5  6
	"""

	def __init__(self):
		self.players = array.array('Q', [0, 0]) # White and black bitboards
		self.turn = 0

		self.last_pos = -1 # Used for score

	def __repr__(self):
		x, o = self.players[0], self.players[1]
		s = ''
		for r in reversed(range(6)):
			for c in range(7):
				if bb.get_bit(x, r*7+c):
					s += 'X '
				elif bb.get_bit(o, r*7+c):
					s += 'O '
				else:
					s += '- '
			s += '\n'
		s += '0 1 2 3 4 5 6'
		return s

	def gen_actions(self):
		empty = ~(self.players[0] | self.players[1])
		actions = []
		for i in range(7):
			if empty & columns[i]:
				actions.append(i)
		return actions
	
	def make(self, move):
		empty = ~(self.players[0] | self.players[1])
		slot = bb.first_bit(empty & columns[move])
		self.players[self.turn] |= slot
		self.turn = 1 - self.turn

		self.last_pos = bb.last_index(slot)

	def unmake(self, move):
		self.turn = 1 - self.turn
		pieces = self.players[self.turn] & columns[move]
		self.players[self.turn] &= ~bb.last_bit(pieces)
	
	def score(self):
		""" returns (terminal?, score)
		"""
		notplayer = ~self.players[1 - self.turn]
		for line in lines_including[self.last_pos]:
			if not notplayer & line:
				return True, -1
		if full & ~(self.players[0] | self.players[1]):
			return False, 0
		return True, 0

def perft(state, depth):
	count = 0
	for move in state.gen_moves():
		if depth > 1:
			state.make(move)
			terminal, _ = state.score()
			if terminal:
				count += 1
			else:
				count += perft(state, depth-1)
			state.unmake(move)
		else:
			count += 1
	return count
"""
import random
def rollout(state):
	terminal, score = state.score()
	if terminal:
		return score
	moves = list(state.gen())
	rand_move = random.choice(moves)
	state.make(rand_move)
	return -rollout(state)

x_win, y_win, draw = 0, 0, 0

for i in range(100000):
	s = State()
	r = rollout(s)
	if r == 1:
		x_win += 1
	elif r == -1:
		y_win += 1
	else:
		draw += 1

print(x_win, y_win, draw)"""