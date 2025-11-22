class Automaton:
    def __init__(self, N_state, state):
        assert N_state % 2 == 0, "N_state must be even"

        self.N_state = N_state
        self.middle_state = N_state // 2

        self.state = state
        self.action = self._action()

    def _action(self):
        return 1 if (self.state > self.middle_state) else 0

    def reward(self):
        changed = False
        if self.state < self.N_state:
            changed = (self.state == self.middle_state)
            self.state += 1
            if changed:
                self.action ^= 1
        return changed

    def penalty(self):
        changed = False
        if self.state > 1:
            changed = (self.state == (self.middle_state + 1))
            self.state -= 1
            if changed:
                self.action ^= 1
        return changed

    def update(self):
        self.action = self._action()
