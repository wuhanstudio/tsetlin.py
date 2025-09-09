import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from tsetlin import Tsetlin
from tsetlin.utils import booleanize_features

EPOCHS = 10

N_CLAUSE = 100
N_STATE  = 400

iris = pd.read_csv("iris.csv")

iris['label'] = iris['species'].map({
    'setosa': 0,
    'versicolor': 1,
    'virginica': 2
})

y = iris["label"].to_numpy()
X = iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]].to_numpy()

# Normalization
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)

X_bool = booleanize_features(X, X_mean, X_std)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_bool, y, test_size=0.2, random_state=0)

tsetlin = Tsetlin(N_feature=X_train.shape[1], N_class=3, N_clause=N_CLAUSE, N_state=N_STATE)

y_pred = tsetlin.predict(X_test)
accuracy = np.sum(y_pred == y_test) / len(y_test)

for epoch in range(EPOCHS):
    print(f"[Epoch {epoch+1}/{EPOCHS}] Train Accuracy: {accuracy * 100:.2f}%")
    for i in tqdm(range(len(X_train))):
        tsetlin.step(X_train[i], y_train[i], T=30, s=6)

    y_pred = tsetlin.predict(X_train)
    accuracy = np.sum(y_pred == y_train) / len(y_train)

# tsetlin.fit(X_train, y_train, T=15, s=3, epochs=EPOCHS)

# Final evaluation
y_pred = tsetlin.predict(X_test)
accuracy = np.sum(y_pred == y_test) / len(y_test)

print(f"Test Accuracy: {accuracy * 100:.2f}%")

print()

# Save the model
tsetlin.save_model("tsetlin_model.pb", type="training")
print("Model saved to tsetlin_model.pb")

# Load the model
n_tsetlin = Tsetlin(N_feature=X_train.shape[1], N_class=3, N_clause=N_CLAUSE, N_state=N_STATE)
n_tsetlin.load_model("tsetlin_model.pb")
print("Model loaded from tsetlin_model.pb")

print()

# Evaluate the loaded model
n_y_pred = n_tsetlin.predict(X_test)
accuracy = np.sum(y_pred == y_test) / len(y_test)

print(f"Test Accuracy (Loaded Model): {accuracy * 100:.2f}%")