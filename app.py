import os

import pandas as pd
from sklearn.model_selection import train_test_split

import streamlit as st
from stqdm import stqdm
import matplotlib.pyplot as plt

from tsetlin import Tsetlin
from tsetlin.utils import booleanize_features
from tsetlin.utils import mean, std

iris = pd.read_csv("iris.csv")

iris_class = ['setosa', 'versicolor', 'virginica']
iris['label'] = iris['species'].map({
    'setosa': 0,
    'versicolor': 1,
    'virginica': 2
})

y = iris["label"].to_numpy()
X = iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]].to_numpy()

# Normalization
X_mean = mean(X)
X_std = std(X)

X_bool = booleanize_features(X, X_mean, X_std)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_bool, y, test_size=0.2, random_state=0)

@st.fragment
def draw_download_button(filename, label, mime):
    with open(filename, "rb") as file:
        btn = st.download_button(
            label=label,
            data=file,
            type="primary",
            file_name=filename,
            mime=mime,
        )

@st.fragment
def iris_data():
    st.title("Iris Dataset")
    st.dataframe(iris)

@st.fragment
def train():
    st.title("Model Training")

    st.number_input("Number of Epochs", min_value=1, max_value=100, value=10, step=1, key="n_epochs")

    st.number_input("Number of Clauses", min_value=10, max_value=1000, value=100, step=10, key="n_clause")
    st.number_input("Number of States", min_value=100, max_value=1000, value=400, step=10, key="n_state")

    EPOCHS = st.session_state.n_epochs
    N_CLAUSE = st.session_state.n_clause
    N_STATE  = st.session_state.n_state

    if st.button("Train Model", type="primary"):
        tsetlin = Tsetlin(N_feature=X_train.shape[1], N_class=3, N_clause=N_CLAUSE, N_state=N_STATE)

        accuracy_list = []

        for epoch in stqdm(range(EPOCHS), desc=f"Training Model"):
            for i in range(len(X_train)):
                tsetlin.step(X_train[i], y_train[i], T=30, s=6)

            y_pred = tsetlin.predict(X_train)
            accuracy = sum(y_pred == y_train) / len(y_train)

            accuracy_list.append(accuracy * 100)

        fig, ax = plt.subplots()
        ax.set_xticks(range(EPOCHS))
        ax.plot(range(EPOCHS), accuracy_list)

        st.pyplot(fig)

        st.write(f"Train Accuracy: {accuracy * 100:.2f}%")

        # Final evaluation
        y_pred = tsetlin.predict(X_test)
        accuracy = (sum(y_pred == y_test) / len(y_test))

        st.write(f"Test Accuracy: {accuracy * 100:.2f}%")

        tsetlin.save_model("tsetlin_model.pb", type="training")
        draw_download_button("tsetlin_model.pb", "Download model", "application/pb")


@st.fragment
def evaluate():
    st.title("Model Evaluation")

    # Read model file
    f_model = st.file_uploader("Choose model file", type=["pb"])

    if f_model is not None:
        # Save the uploaded model to a temporary file
        if not os.path.exists("temp"):
            os.makedirs("temp")
        f_model_path = os.path.join("temp", f_model.name)
        with open(f_model_path, "wb") as f:
            f.write(f_model.getvalue())

        n_tsetlin = Tsetlin.load_model(f"temp/{f_model.name}")

        # Evaluate the loaded model
        y_pred, votes = n_tsetlin.predict(X_test, return_votes=True)

        accuracy = (sum(y_pred == y_test) / len(y_test))
        st.write(f"Test Accuracy: {accuracy * 100:.2f}%")

        result = pd.DataFrame({
            "True Label": [iris_class[label] for label in y_test],
            "Predicted Label": [iris_class[label] for label in y_pred],
            "Correct": [true == pred for true, pred in zip(y_test, y_pred)]
        })
        st.dataframe(result)

        index = st.number_input("The index of the testing data", 0, len(X_test) - 1)

        st.write(f"True Label: {iris_class[y_test[index]]} - Predicted Label: {iris_class[y_pred[index]]}")

        fig, ax = plt.subplots()
        ax.bar(["Setosa", "Versicolor", "Virginica"], votes[index])
        st.pyplot(fig)
   

if __name__ == "__main__":
    st.header("Tsetlin Machine", divider=True)

    iris_data()

    st.divider()

    train()

    st.divider()

    evaluate()
