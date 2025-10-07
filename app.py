import os

import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

import streamlit as st
from stqdm import stqdm
import matplotlib.pyplot as plt

from tsetlin import Tsetlin
from tsetlin.utils import booleanize_features
from tsetlin.utils import mean, std

from pretty_confusion_matrix import pp_matrix

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
    iris = pd.read_csv("iris.csv")

    iris_class = ['setosa', 'versicolor', 'virginica']
    iris['label'] = iris['species'].map({
        'setosa': 0,
        'versicolor': 1,
        'virginica': 2
    })

    y = iris["label"].to_numpy()
    X = iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]].to_numpy()

    st.title("Iris Dataset")
    st.dataframe(iris)

    return X, y, iris_class

@st.fragment
def train(X_train, y_train):
    st.title("Model Training")

    st.number_input("Number of Epochs", min_value=1, max_value=100, value=10, step=1, key="n_epochs")

    st.number_input("Number of Clauses", min_value=10, max_value=1000, value=100, step=10, key="n_clause")
    st.number_input("Number of States", min_value=100, max_value=1000, value=400, step=10, key="n_state")

    EPOCHS = st.session_state.n_epochs
    N_CLAUSE = st.session_state.n_clause
    N_STATE  = st.session_state.n_state

    T = st.slider("Hyperparameter T", 0, N_STATE, 20)
    s = st.slider("Hyperparameter s", 0.0, 10.0, 3.5, 0.1)

    if st.button("Train Model", type="primary"):
        tsetlin = Tsetlin(N_feature=len(X_train[0]), N_class=3, N_clause=N_CLAUSE, N_state=N_STATE)

        accuracy_list = []

        for epoch in stqdm(range(EPOCHS), desc=f"Training Model"):
            for i in range(len(X_train)):
                tsetlin.step(X_train[i], y_train[i], T=T, s=s)

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
def evaluate(X_test, y_test, iris_class):
    st.title("Model Evaluation")

    st.write(f"Number of Features: {len(X_test[0])}")

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

        if n_tsetlin.n_features != len(X_test[0]):
            st.error(f"Uploaded model feature size {n_tsetlin.n_features} does not match test feature size {len(X_test[0])}")
            return

        # Evaluate the loaded model
        y_pred, votes = n_tsetlin.predict(X_test, return_votes=True)

        accuracy = (sum(y_pred == y_test) / len(y_test))
        st.write(f"Test Accuracy: {accuracy * 100:.2f}%")

        cm = metrics.confusion_matrix(y_test, y_pred)
        df_cm = pd.DataFrame(cm, index=range(0, 3), columns=range(0, 3))
        st.pyplot(pp_matrix(df_cm))

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

    X, y, iris_class = iris_data()

    st.title("Data Preprocessing")
    st.write("Booleanization of features")

    st.selectbox("Number of Bits", options=[1, 2, 4, 8], index=3, key="n_bit")
    N_BIT = st.session_state.n_bit

    # Normalization
    X_mean = mean(X)
    X_std = std(X)

    X_bool = booleanize_features(X, X_mean, X_std, num_bits=N_BIT)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_bool, y, test_size=0.2, random_state=0)

    st.divider()
    
    train(X_train, y_train)

    st.divider()

    evaluate(X_test, y_test, iris_class)
