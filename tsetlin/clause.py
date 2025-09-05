import numpy as np
from tsetlin.automaton import Automaton

class Clause:
    def __init__(self, N_feature, N_state):

        assert N_state % 2 == 0, "N_state must be even"

        self.N_feature = N_feature
        self.N_literals = 2 * N_feature

        # For each feature, initialize two automata (X and NOT X)
        self.automata = [Automaton(N_state, -1) for _ in range(2 * N_feature)]

        # Randomly initialize automata states middle_state + {0,1}
        for i in range(self.N_feature):
            choice = np.random.choice([0, 1])
            self.automata[2 * i].state = N_state // 2 + choice
            self.automata[2 * i + 1].state = N_state // 2 + (1 - choice)

    def evaluate(self, X):
        output = 1
        for i in range(self.N_feature):
            if self.automata[2 * i].action() == 0 and X[i] == 1:
                output = 0
                break
            if self.automata[2 * i + 1].action() == 0 and X[i] == 0:
                output = 0
                break
        return output

    def update(self, X, match_target, clause_output, s=3):
        # Type I Feedback (Recognize patterns)
        if match_target == 1:
            s1 = 1 / s
            s2 = (s - 1) / s

            if clause_output == 0:
                for i in range(self.N_feature):
                    # TODO: Do we need rand() twice
                    if np.random.rand() <= s1:
                        self.automata[2 * i].penalty()
                        self.automata[2 * i + 1].penalty()

            if clause_output == 1:
                for i in range(self.N_feature):
                    if X[i] != self.automata[2 * i].action():
                        if np.random.rand() <= s1:
                            self.automata[2 * i].penalty()
                            self.automata[2 * i + 1].reward()
                    else:
                        if np.random.rand() <= s2:
                            self.automata[2 * i].reward()
                            self.automata[2 * i + 1].penalty()

        # Type II Feedback (Erase / Forget patterns)
        elif match_target == 0 and clause_output == 1:
            for i in range(self.N_feature):
                if X[i] != self.automata[2 * i].action():
                    self.automata[2 * i].penalty()
                else:
                    self.automata[2 * i + 1].penalty()
