import numpy as np
from tsetlin.automaton import Automaton

class Clause:
    def __init__(self, N_feature, N_state):

        assert N_state % 2 == 0, "N_state must be even"

        self.N_feature = N_feature
        self.N_literals = 2 * N_feature

        # Positive and Negative Automata for each feature
        self.p_automata = [Automaton(N_state, -1) for _ in range(N_feature)]
        self.n_automata = [Automaton(N_state, -1) for _ in range(N_feature)]
        
        # Randomly initialize automata states middle_state + {0,1}
        for i in range(self.N_feature):
            choice = np.random.choice([0, 1])
            self.p_automata[i].state = N_state // 2 + choice
            self.n_automata[i].state = N_state // 2 + (1 - choice)

    def evaluate(self, X):
        output = 1
        for i in range(self.N_feature):
            # Include positive literal,  but feature is 0
            if self.p_automata[i].action() == 1 and X[i] == 0:
                output = 0
                break
            # Include negative literal, but feature is 1
            if self.n_automata[i].action() == 1 and X[i] == 1:
                output = 0
                break
        return output

    def update(self, X, match_target, clause_output, s):
        # Type I Feedback (Recognize patterns)
        # Want clause_output to be 1
        if match_target == 1:
            s1 = 1 / s
            s2 = (s - 1) / s

            # Erase Pattern
            # Reduce the number of included literals
            if clause_output == 0:
                for i in range(self.N_feature):
                    if np.random.rand() <= s1:
                        self.p_automata[i].penalty()
                        self.n_automata[i].penalty()

            # Recognize Pattern
            # Increase the number of included literals
            if clause_output == 1:
                for i in range(self.N_feature):
                    # Positive literal X
                    if X[i] != self.p_automata[i].action():
                        # B = 0
                        if np.random.rand() <= s1:
                            self.p_automata[i].penalty()
                    else:
                        # B = 1
                        if np.random.rand() <= s2:
                            self.p_automata[i].reward()

                    # Negative literal NOT X
                    if X[i] != (1 - self.n_automata[i].action()):
                        # B = 0
                        if np.random.rand() <= s1:
                            self.n_automata[i].penalty()
                    else:
                        # B = 1
                        if np.random.rand() <= s2:
                            self.n_automata[i].reward()

        # Type II Feedback (Reject Patterns)
        # Want clause_output to be 0
        elif match_target == 0:
            if (clause_output == 1):
                for i in range(self.N_feature):
                    # Positive literal X
                    if X[i] != self.p_automata[i].action():
                        # B = 0 
                        self.p_automata[i].reward()
                    # Negative literal NOT X
                    if X[i] != (1 - self.n_automata[i].action()):
                        # B = 0 
                        self.n_automata[i].reward()
