class Automaton:
    def __init__(self, N_state, state):
        assert N_state % 2 == 0, "N_state must be even"

        self.N_state = N_state
        self.state = state

        self.action = self._action()

    def _action(self):
        return 1 if (self.state > (self.N_state // 2)) else 0

    def reward(self):
        if self.state < self.N_state:
            self.state += 1
            self.action = self._action()
    def penalty(self):
        if self.state > 1:
            self.state -= 1
            self.action = self._action()
    
    def update(self):
        self.action = self._action()
