""" Some helper functions for dealing with bitboards
"""

def make_rep(rows, cols):
    def to_string(bb):
        s = ''
        for r in reversed(range(rows)):
            for c in range(cols):
                if get_bit(bb, r * cols + c):
                    s += '#'
                else:
                    s += '-'
            s += '\n'
        return s[:-1]
    return to_string

def from_pos(pos_list):
    bb = 0
    for p in pos_list:
        bb |= 1 << p
    return bb

def get_bit(bb, pos):
    return bb & (1 << pos)

def first_bit(bb):
    return bb & -bb

def first_index(bb):
    return first_bit(bb).bit_length() - 1

def last_bit(bb):
    return 1 << last_index(bb)

def last_index(bb):
    return bb.bit_length() - 1