# Clause implementation using ONLY integer states (no Automaton objects)
# This version is fully compatible with the new automaton_integer_version module.

import random

# Completely remove Automaton class.
# This module now only provides helper functions operating on integer states.

class Clause:
    def __init__(self, N_feature, N_state):
        assert N_state % 2 == 0, "N_state must be even"

        self.N_feature = N_feature
        self.N_states = N_state
        self.middle = N_state // 2
        self.N_literals = 2 * N_feature

        # Each literal is represented by an integer state
        # Positive and Negative literal states
        self.p_automata = [self.middle for _ in range(N_feature)]
        self.n_automata = [self.middle for _ in range(N_feature)]

        # Randomly initialize to middle + {0,1}
        for i in range(N_feature):
            c = random.randint(0, 1)
            self.p_automata[i] = self.middle + c
            self.n_automata[i] = self.middle + (1 - c)

        self.compress()

    def compress(self, threshold=-1):
        # Included literals: state > middle
        self.p_included_literals = [i for i, s in enumerate(self.p_automata) if s > self.middle]
        self.n_included_literals = [i for i, s in enumerate(self.n_automata) if s > self.middle]

        if threshold > 0:
            # Trainable literals: close to boundary
            self.p_trainable_literals = [i for i, s in enumerate(self.p_automata)
                                         if abs(s - self.middle) <= threshold]
            self.n_trainable_literals = [i for i, s in enumerate(self.n_automata)
                                         if abs(s - self.middle) <= threshold]

    def evaluate(self, X):
        for i in self.p_included_literals:
            if X[i] == 0:
                return 0
        for i in self.n_included_literals:
            if X[i] == 1:
                return 0
        return 1

    def update(self, X, match_target, clause_output, s, threshold=-1, logger=None):
        feedback_count = 0
        s1 = 1 / s
        s2 = (s - 1) / s

        # -------------------------------------------------
        # Type I Feedback
        # -------------------------------------------------
        if match_target == 1:
            # Case 1: Clause output is 0 (Erase Pattern)
            if clause_output == 0:
                targets = range(self.N_feature) if threshold < 0 else self.p_trainable_literals
                for i in targets:
                    # Positive literal penalty
                    if self.p_automata[i] > 1 and random.random() <= s1:
                        self.p_automata[i] -= 1
                        changed = (self.p_automata[i] == self.middle)
                        if changed and i in self.p_included_literals:
                            self.p_included_literals.remove(i)
                        feedback_count += 1

                targets = range(self.N_feature) if threshold < 0 else self.n_trainable_literals
                for i in targets:
                    # Negative literal penalty
                    if self.n_automata[i] > 1 and random.random() <= s1:
                        self.n_automata[i] -= 1
                        changed = (self.n_automata[i] == self.middle)
                        if changed and i in self.n_included_literals:
                            self.n_included_literals.remove(i)
                        feedback_count += 1

            # Case 2: Clause output is 1 (Recognize Pattern)
            else:
                targets = range(self.N_feature) if threshold < 0 else self.p_trainable_literals
                for i in targets:
                    # X literal
                    if X[i] == 1:
                        # Reward positive literal
                        if self.p_automata[i] < self.N_states and random.random() <= s2:
                            self.p_automata[i] += 1
                            changed = (self.p_automata[i] == self.middle + 1)
                            if changed:
                                self.p_included_literals.append(i)
                            feedback_count += 1
                        # Penalty negative literal
                        if self.n_automata[i] > 1 and random.random() <= s1:
                            self.n_automata[i] -= 1
                            changed = (self.n_automata[i] == self.middle)
                            if changed and i in self.n_included_literals:
                                self.n_included_literals.remove(i)
                            feedback_count += 1

                    # NOT X literal
                    else:
                        # Reward negative literal
                        if self.n_automata[i] < self.N_states and random.random() <= s2:
                            self.n_automata[i] += 1
                            changed = (self.n_automata[i] == self.middle + 1)
                            if changed:
                                self.n_included_literals.append(i)
                            feedback_count += 1
                        # Penalty positive literal
                        if self.p_automata[i] > 1 and random.random() <= s1:
                            self.p_automata[i] -= 1
                            changed = (self.p_automata[i] == self.middle)
                            if changed and i in self.p_included_literals:
                                self.p_included_literals.remove(i)
                            feedback_count += 1

        # -------------------------------------------------
        # Type II Feedback (Reject Pattern)
        # -------------------------------------------------
        else:
            if clause_output == 1:
                targets = range(self.N_feature) if threshold < 0 else self.p_trainable_literals
                for i in targets:
                    # Add positive literal
                    if X[i] == 0 and self.p_automata[i] <= self.middle:
                        self.p_automata[i] += 1
                        changed = (self.p_automata[i] == self.middle + 1)
                        if changed:
                            self.p_included_literals.append(i)
                        feedback_count += 1

                targets = range(self.N_feature) if threshold < 0 else self.n_trainable_literals
                for i in targets:
                    # Add negative literal
                    if X[i] == 1 and self.n_automata[i] <= self.middle:
                        self.n_automata[i] += 1
                        changed = (self.n_automata[i] == self.middle + 1)
                        if changed:
                            self.n_included_literals.append(i)
                        feedback_count += 1

        return feedback_count

    # -------------------------------------------------
    # Export concatenated states (positive + negative)
    # -------------------------------------------------
    def get_state(self):
        return self.p_automata + self.n_automata

    # -------------------------------------------------
    # Set states from array
    # -------------------------------------------------
    def set_state(self, states, threshold=-1):
        assert len(states) == 2 * self.N_feature
        self.p_automata = states[: self.N_feature]
        self.n_automata = states[self.N_feature :]
        self.compress(threshold=threshold)
