import random
import numpy as np
from tqdm import tqdm

from tsetlin.clause import Clause

class Tsetlin:
    def __init__(self, N_feature, N_class, N_clause, N_state):

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

        # Update clauses for the target class
        for i in range(int(self.n_clauses / 2)):
            if (np.random.rand() <= c1):
                # Positive Clause: Type I Feedback
                self.pos_clauses[y_target][i].update(X, 1, self.pos_clauses[y_target][i].evaluate(X), s=s)
            if (np.random.rand() <= c1):
                # Negative Clause: Type II Feedback
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
                # Positive Clause: Type II Feedback
                self.pos_clauses[other_class][i].update(X, 0, self.pos_clauses[other_class][i].evaluate(X), s=s)
            if (np.random.rand() <= c2):
                # Negative Clause: Type I Feedback
                self.neg_clauses[other_class][i].update(X, 1, self.neg_clauses[other_class][i].evaluate(X), s=s)

    def fit(self, X, y, T, s, epochs):
        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            for i in range(len(X)):
                self.step(X[i], y[i], T=T, s=s)

    def load_model(self, path):
        import tsetlin_pb2
        tm = tsetlin_pb2.Tsetlin()

        with open(path, "rb") as f:
            tm.ParseFromString(f.read())

        self.n_classes = tm.n_class
        self.n_features = tm.n_feature
        self.n_clauses = tm.n_clause
        self.n_states = tm.n_state
    
        self.pos_clauses = []
        self.neg_clauses = []
        for i in range(self.n_classes):
            pos_clauses = []
            neg_clauses = []
            for j in range(self.n_clauses // 2):
                p_clause = tm.clauses[i * self.n_clauses + j * 2]
                n_clause = tm.clauses[i * self.n_clauses + j * 2 + 1]

                # Set positive clauses
                pos_clause = Clause(self.n_features, self.n_states)
                pos_clause.set_state(np.array(p_clause.data).astype(np.uint32))
                pos_clauses.append(pos_clause)

                # Set negative clauses
                neg_clause = Clause(self.n_features, self.n_states)
                neg_clause.set_state(np.array(n_clause.data).astype(np.uint32))
                neg_clauses.append(neg_clause)

            self.pos_clauses.append(pos_clauses)
            self.neg_clauses.append(neg_clauses)

    def save_model(self, path, type="training"):
        import tsetlin_pb2
        tm = tsetlin_pb2.Tsetlin()

        tm.n_class = self.n_classes
        tm.n_feature = self.n_features
        tm.n_clause = self.n_clauses
        tm.n_state = self.n_states

        if type not in ["training", "inference"]:
            raise ValueError("type must be either 'training' or 'inference'")
        
        if type == "training":
            tm.model_type = tsetlin_pb2.Tsetlin.ModelType.TRAINING
        else:
            tm.model_type = tsetlin_pb2.Tsetlin.ModelType.INFERENCE

        for i in range(self.n_classes):
            for j in range(self.n_clauses // 2):
                # Positive clauses
                pos_c = tsetlin_pb2.Clause()
                pos_c.n_feature = self.n_features
                pos_c.n_state = self.n_states

                pos_c.data.extend(self.pos_clauses[i][j].get_state().flatten().tolist())

                tm.clauses.append(pos_c)

                # Negative clauses
                neg_c = tsetlin_pb2.Clause()
                neg_c.n_feature = self.n_features
                neg_c.n_state = self.n_states

                neg_c.data.extend(self.neg_clauses[i][j].get_state().flatten().tolist())

                tm.clauses.append(neg_c)

        with open(path, "wb") as f:
            f.write(tm.SerializeToString())
