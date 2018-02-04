import nash
import random

for _ in range(20):
    s = nash.NashState()
    while s.get_score() == -1:
        a = random.choice(s.gen_actions())
        s.do_action(a)
    print(s)
    print(s.turn)
    print(s.nodes)