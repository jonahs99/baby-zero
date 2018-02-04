class Model:
    def __init__(self, net):
        self.net = net
    def represent(self, state):
        raise(NotImplementedError)