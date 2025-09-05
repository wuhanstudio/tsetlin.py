import random
import numpy as np
from tqdm import tqdm

from tsetlin.clause import Clause

class Tsetlin:
    def __init__(self, N_feature, N_class, N_clause=100, N_state=400):

        assert N_state % 2 == 0, "N_state must be even"
        assert N_clause % 2 == 0, "N_clause must be even"

        self.num_features = N_feature
        self.num_classes = N_class
        self.num_clauses = N_clause
        self.num_states = N_state

        self.clauses = []
        for _ in range(N_class):
            self.clauses.append([Clause(N_feature, N_state=N_state) for _ in range(N_clause)])

    def predict(self, X):
        y_pred = []
        for i in range(len(X)):
            votes = np.zeros(self.num_classes)
            for c in range(self.num_classes):
                for j, clause in enumerate(self.clauses[c]):
                    sign = 1
                    # sign = 1 - 2 * (j % 2)  # Odd clauses are negative
                    
                    clause_output = clause.evaluate(X[i])
                    votes[c] += (clause_output * sign)
            y_pred.append(np.argmax(votes))
        return np.array(y_pred)

    def step(self, X, y_target, T=15, s=3):
        for c in range(self.num_classes):
            class_sum = 0
            for clause in self.clauses[c]:
                class_sum += clause.evaluate(X)
            c1 = (T - class_sum) / (2 * T)
            c2 = (T + class_sum) / (2 * T)

            for clause in self.clauses[c]:
                clause_output = clause.evaluate(X)
                if y_target == c and (np.random.rand() <= c1):
                    clause.update(X, 1, clause_output, s=s)
                elif y_target != c and (np.random.rand() <= c2):
                    clause.update(X, 0, clause_output, s=s)

    def fit(self, X, y, epochs=10):
        for epoch in tqdm(range(epochs)):
            for i in range(len(X)):
                self.step(X[i], y[i])
