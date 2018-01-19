import random

def choice(options, probs):
    r = random.random()
    c = 0
    i = -1
    while c < r:
        c += probs[i]
        i += 1
        if i == len(options) - 1:
            return options[i]
    return options[i]