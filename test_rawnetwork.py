from connect4_policy import Connect4Policy
from connect4 import Connect4State

state = Connect4State()
policy = Connect4Policy()

while state.get_score() == -1:
    actions = state.gen_actions()
    dist = policy.predict(state, actions)
    
    options = list(zip(actions, dist))

    action = max(options, key=lambda option: option[1])[0]
    state.do_action(action)

    for option in options:
        print(option)
    print(state)