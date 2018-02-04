class State:
    def __init__(self):
        pass

    def gen_actions(self):
        raise(NotImplementedError)

    def do_action(self):
        raise(NotImplementedError)

    def undo_action(self):
        raise(NotImplementedError)

    def get_score(self):
        raise(NotImplementedError)
        
    def __repr__(self):
        raise(NotImplementedError)