import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from tsetlin import Tsetlin
from tsetlin.utils import booleanize_features

EPOCHS = 10

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

tsetlin = Tsetlin(N_feature=X_train.shape[1], N_class=3, N_clause=100)

y_pred = tsetlin.predict(X_test)
accuracy = np.sum(y_pred == y_test) / len(y_test)

for epoch in range(EPOCHS):
    print(f"[Epoch {epoch+1}/{EPOCHS}] Accuracy: {accuracy * 100:.2f}%")
    pbar = tqdm(enumerate(X_train), total=len(X_train))
    for i, x in pbar:
        tsetlin.step(x, y_train[i])

        y_pred = tsetlin.predict(X_test)
        accuracy = np.sum(y_pred == y_test) / len(y_test)

        pbar.desc = f'Accuracy {accuracy * 100:.2f}%'
    pbar.close()

# tsetlin.fit(X_train, y_train, epochs=EPOCHS)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
