import random

def choice(probs):
    r = random.random()
    c = 0
    i = 0
    while c < r:
        print(c,r)
        c += probs[i]
        i += 1
        if i == len(probs) - 1:
            return i
    return i