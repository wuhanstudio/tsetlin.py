import random
import numpy as np
from tqdm import tqdm

from tsetlin.clause import Clause

class Tsetlin:
    def __init__(self, N_feature, N_class, N_clause=100, N_state=400):

        assert N_state % 2 == 0, "N_state must be even"
        assert N_clause % 2 == 0, "N_clause must be even"

        self.n_features = N_feature
        self.n_classes = N_class

        self.n_clauses = N_clause
        self.n_states = N_state

        self.pos_clauses = []
        self.neg_clauses = []
        for _ in range(N_class):
            self.pos_clauses.append([Clause(N_feature, N_state=N_state) for _ in range(int(N_clause / 2))])
            self.neg_clauses.append([Clause(N_feature, N_state=N_state) for _ in range(int(N_clause / 2))])

    def predict(self, X):
        y_pred = []
        for i in range(len(X)):
            votes = np.zeros(self.n_classes)
            for c in range(self.n_classes):
                for j in range(int(self.n_clauses / 2)):
                    votes[c] += self.pos_clauses[c][j].evaluate(X[i])
                    votes[c] -= self.neg_clauses[c][j].evaluate(X[i])
            y_pred.append(np.argmax(votes))
        return np.array(y_pred)

    def step(self, X, y_target, T, s):
        # Pair-wise learning

        # Pair 1: Target class
        class_sum = 0
        for i in range(int(self.n_clauses / 2)):
            class_sum += self.pos_clauses[y_target][i].evaluate(X)
            class_sum -= self.neg_clauses[y_target][i].evaluate(X)

        # Clamp class_sum to [-T, T]
        class_sum = np.clip(class_sum, -T, T)
    
        # Calculate probabilities
        c1 = (T - class_sum) / (2 * T)

        # Update clauses for class c
        for i in range(int(self.n_clauses / 2)):
            if (np.random.rand() <= c1):
                # Type I Feedback
                self.pos_clauses[y_target][i].update(X, 1, self.pos_clauses[y_target][i].evaluate(X), s=s)
            if (np.random.rand() <= c1):
                # Type II Feedback
                self.neg_clauses[y_target][i].update(X, 0, self.neg_clauses[y_target][i].evaluate(X), s=s)

        # Pair 2: Non-target classes
        other_class = random.choice([x for x in range(self.n_classes) if x != y_target])

        class_sum = 0
        for i in range(int(self.n_clauses / 2)):
            class_sum += self.pos_clauses[other_class][i].evaluate(X)
            class_sum -= self.neg_clauses[other_class][i].evaluate(X)

        # Clamp class_sum to [-T, T]
        class_sum = np.clip(class_sum, -T, T)

        # Calculate probabilities
        c2 = (T + class_sum) / (2 * T)
        for i in range(int(self.n_clauses / 2)):
            if (np.random.rand() <= c2):
                # Type II Feedback
                self.pos_clauses[other_class][i].update(X, 0, self.pos_clauses[other_class][i].evaluate(X), s=s)
            if (np.random.rand() <= c2):
                # Type I Feedback
                self.neg_clauses[other_class][i].update(X, 1, self.neg_clauses[other_class][i].evaluate(X), s=s)

    def fit(self, X, y, T, s, epochs=10):
        for epoch in tqdm(range(epochs)):
            for i in range(len(X)):
                self.step(X[i], y[i], T=T, s=s)
