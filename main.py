import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tsetlin.utils.booleanize import boolieanize
from tsetlin.tsetlin import Tsetlin

def boolieanize_features(X, mean, std):
    # Normalization to [0, 255]
    X = (X - mean) / std
    X = np.array(X * 255, dtype=np.uint8)

    X_bool = []
    for x_features in X:
        bool_features = []
        for x in x_features:
            bool_features.append(boolieanize(x))
        X_bool.append(np.concatenate(bool_features))

    return np.array(X_bool, dtype=bool)

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

X_bool = boolieanize_features(X, X_mean, X_std)

X_train, X_test, y_train, y_test = train_test_split(X_bool, y, test_size=0.2, random_state=0)

tsetlin = Tsetlin(N_feature=X_train.shape[1], N_class=3, N_clause=50)
tsetlin.fit(X_train, y_train, epochs=200)

y_pred = tsetlin.predict(X_test)
accuracy = np.sum(y_pred == y_test) / len(y_test)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
