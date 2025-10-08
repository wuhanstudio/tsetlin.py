# Tsetlin Machine

> The Iris Dataset

## Todo List

[●] Debug: Import / Save model  
[●] Debug: Print clause output & class sum  
<br/>
[●] MNIST Dataset  
[●] Interface for quick benchmark (T, s, confusion matrix)  
[●] Different booleanize methods  
<br/>
[●] TMU Library  
[●] Hyper-parameter (Optuna)  
<br/>
[●] [**UK DALE - Preprocessing**](./dataset/)  
[●] [**Online / Offline Data Simulator**](./dataset/simulator)  
<br/>
[&nbsp; ] NILM (FHMM / CO / KNN / SVM / DNN)  
[&nbsp; ] **Unsupervised TM using ART**  
<br/>
[&nbsp; ] Improve: [Adaptive Reasonance](https://arxiv.org/pdf/1905.11437)  
[&nbsp; ] Improve: [Coelesed TM with Clause Sharing](https://arxiv.org/abs/2108.07594)  

## Prerequisites

Using `pip`:

```
$ python -m pip install -r requirements.txt
```

Using `pipenv`:

```
$ python -m pip install pipenv

$ pipenv install
$ pipenv shell
```

Using `uv`

```
$ uv python install 3.12
$ uv venv --python 3.12
$ uv sync

# Linux
$ . .venv/bin/activate

# Windows
$ Set-ExecutionPolicy Unrestricted -Scope Process
$ .venv\Scripts\activate

```

## Quick Start

Train and save the model

```
$ python main.py
```

Using `streamlit` as the UI:

```
$python main_ui.py
```

![](demo.png)

## Optuna

```
$ python main.py --optuna
```

```
$ optuna-dashboard sqlite:///db.sqlite3
```

## TMU

> [!NOTE]  
> For Windows users, Visual Studio (Microsoft C++ Build Tools) is needed to compile `TMU` package.

Using `pipenv`:

```
$ pipenv install git+https://github.com/cair/tmu.git
```

Using `uv`:

```
$ uv pip install git+https://github.com/cair/tmu.git
```

## Pytest

```
$ pytest
```

```
============================================= test session starts =============================================
platform win32 -- Python 3.10.18, pytest-8.4.2, pluggy-1.6.0
rootdir: C:\Users\Han\Desktop\tsetlin.py
collected 7 items                                                                                              

tests\test_automaton.py ....                                                                             [ 57%]
tests\test_booleanize.py .                                                                               [ 71%] 
tests\test_clause.py ..                                                                                  [100%]

============================================== 7 passed in 0.36s ============================================== 
```
