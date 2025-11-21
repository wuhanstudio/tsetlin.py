import numpy as np
import pandas as pd
from loguru import logger

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tsetlin import Tsetlin
from tsetlin.utils import booleanize_features
from tsetlin.utils.tqdm import m_tqdm
from tsetlin.utils.log import log

from pyinstrument import Profiler

N_BIT = 8  # Number of bits for booleanization
N_EPOCHS = 10  # Number of training epochs

TRAIN_BUILDING = [1, 2, 3]
TEST_BUIULDING = [5]

def balance_dataset(X_train, y_train, num_per_class=1000):
    indices = []

    for cls in np.unique(y_train):
        cls_idx = np.where(y_train == cls)[0]
        chosen = np.random.choice(cls_idx, num_per_class, replace=False)
        indices.extend(chosen)

    indices = np.array(indices)
    np.random.shuffle(indices)
    return indices

train_files = [f"building_{building}_main_transients_train.csv" for building in TRAIN_BUILDING]
test_files = [f"building_{building}_main_transients_train.csv" for building in TEST_BUIULDING]

# Read and concatenate
train_df = pd.concat((pd.read_csv(f"model/{f}") for f in train_files), ignore_index=True)
test_df = pd.concat((pd.read_csv(f"model/{f}") for f in test_files), ignore_index=True)

X_train = train_df[["transition", "duration"]]
X_test = test_df[["transition", "duration"]]

y_train = train_df["fridge_label"]
y_test = test_df["fridge_label"]

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Normalization
X_mean = pd.concat([X_train, X_test]).mean().to_list()
X_std = pd.concat([X_train, X_test]).std().to_list()

# Balance the training dataset
max_num = min(np.bincount(y_train))

indices = balance_dataset(X_train, y_train, num_per_class=max_num)
X_train = X_train.to_numpy()[indices]
y_train = y_train[indices]

# Balance the testing dataset
max_num = min(np.bincount(y_test))
indices = balance_dataset(X_test, y_test, num_per_class=max_num)
X_test = X_test.to_numpy()[indices]
y_test = y_test[indices]

logger.info(f"Train images shape: {X_train.shape}, Train labels shape: {y_train.shape}")
logger.info(f"Test images shape: {X_test.shape}, Test labels shape: {y_test.shape}")

X_train = booleanize_features(X_train, X_mean, X_std, num_bits=N_BIT)
X_test = booleanize_features(X_test, X_mean, X_std, num_bits=N_BIT)

def objective(trial):
    n_state = trial.suggest_int("n_state", 2, 100, step=2)
    n_clause = trial.suggest_int("n_clause", 2, 100, step=2)

    T = trial.suggest_int("T", 1, n_state)
    s = trial.suggest_float("s", 1.0, 10.0, step=0.1)

    tsetlin = Tsetlin(N_feature=len(X_train[0]), N_class=2, N_clause=n_clause, N_state=n_state)

    for epoch in range(N_EPOCHS):
        for i in m_tqdm(range(len(X_train))):
            tsetlin.step(X_train[i], y_train[i], T=T, s=s)

    y_pred = tsetlin.predict(X_test)
    
    accuracy = sum([ 1 if pred == test else 0 for pred, test in zip(y_pred, y_test)]) / len(y_test)

    return (1.0 - accuracy)

if False:
    import optuna
    
    # Create a new study.
    # study = optuna.create_study()

    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
        study_name="tsetlin-machine-redd",
        load_if_exists="True"
    )
    
    # Invoke optimization of the objective function.
    study.optimize(objective, n_trials=100)  

    print(f"Best value: {study.best_value} (params: {study.best_params})")
else:
    N_CLAUSE = 200  # Number of clauses
    N_STATE = 100  # Number of states per automaton
    T = 100  # Threshold
    s = 5.0  # Specificity

    log(f"Using {N_BIT} bits for booleanization")
    log(f"Number of clauses: {N_CLAUSE}, Number of states: {N_STATE}")
    log(f"Threshold T: {T}, Specificity s: {s}")

    log(f"Feature means: {X_mean}")
    log(f"Feature stds: {X_std}")

    tsetlin = Tsetlin(N_feature=len(X_train[0]), N_class=2, N_clause=N_CLAUSE, N_state=N_STATE)

    y_pred = tsetlin.predict(X_test)
    accuracy = sum(y_pred == y_test) / len(y_test)

    profiler = Profiler()
    profiler.start()

    for epoch in range(N_EPOCHS):
        log(f"[Epoch {epoch+1}/{N_EPOCHS}] Train Accuracy: {accuracy * 100:.2f}%")
        for i in m_tqdm(range(len(X_train))):
            tsetlin.step(X_train[i], y_train[i], T=T, s=s)

        tsetlin.freeze()
        y_pred = tsetlin.predict(X_train)
        tsetlin.unfreeze()

        accuracy = sum(y_pred == y_train) / len(y_train)

    profiler.stop()
    profiler.print()

    log("")

    # Final evaluation
    y_pred = tsetlin.predict(X_test)
    accuracy = sum(y_pred == y_test) / len(y_test)

    log(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Save the model
    model_name = "tsetlin_model_redd"
    tsetlin.save_model(f"{model_name}.pb", type="training")
    tsetlin.save_umodel(f"{model_name}.upb")
    log(f"Model saved to {model_name}.pb")

    log("")

    # Load the model
    n_tsetlin = Tsetlin.load_model(f"{model_name}.pb")
    log(f"Model loaded from {model_name}.pb")

    # Load the micropython model
    # n_tsetlin = Tsetlin.load_umodel(f"{model_name}.upb")
    # log(f"Model loaded from {model_name}.upb")

    # Evaluate the loaded model
    n_y_pred = n_tsetlin.predict(X_test)
    accuracy = sum([ 1 if pred == test else 0 for pred, test in zip(n_y_pred, y_test)]) / len(y_test)

    log(f"Test Accuracy (Loaded Model): {accuracy * 100:.2f}%")

# Random Forest Classifier
# from sklearn.ensemble import RandomForestClassifier

# print("Random Forest Classifier Results:")
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)
# y_pred = rf_model.predict(X_test)

# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# Decision Tree Classifier
# from sklearn.tree import DecisionTreeClassifier

# print("Decision Tree Classifier Results:")
# dt_model = DecisionTreeClassifier(random_state=42)
# dt_model.fit(X_train, y_train)
# y_pred = dt_model.predict(X_test)

# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# Support Vector Classifier
# from sklearn.svm import SVC

# print("Support Vector Classifier Results:")
# svc_model = SVC(kernel='linear', random_state=42)
# svc_model.fit(X_train, y_train)
# y_pred = svc_model.predict(X_test)

# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
