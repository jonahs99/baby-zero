def play(state_type, agent1, agent2, label=None):
    state = state_type()
    agent1.reset()
    agent2.reset()
    while not state.score()[1]:
        print(state.turn)
        player = (agent1, agent2)[state.turn]
        action = player.action(200)
        agent1.do_action(action)
        agent2.do_action(action)
        state.do_action(action)
        print(state)
    print(1 - state.turn, 'wins')
    if label:
        print(label, 'done.')
    return 1 - state.turn

import agent
import examples.connect4 as connect4
import torch

from multiprocessing import Pool

model = connect4.Model()
model.load_state_dict(torch.load('./examples/data/model.pt'))

a1 = agent.TrainAgent(connect4.State(), model)
a1.mcts.eps = 1

a2 = agent.TrainAgent(connect4.State(), model)
a2.mcts.eps = 0.50

pool = Pool(processes=1)
results12 = pool.starmap(play, [(connect4.State, a1, a2, i) for i in range(10)])
results21 = pool.starmap(play, [(connect4.State, a2, a1, i) for i in range(10)])

print(results12, results21)

#play(connect4.State, a2, a1)