# Tsetlin Machine

> The Iris Dataset

## Todo List

[●] Debug: Import / Save model  
[&nbsp; ] Debug: Print clause output & class sum

[&nbsp; ] Improve: **Unsupervised TM**  
[&nbsp; ] Improve: [Coelesed TM with Clause Sharing](https://arxiv.org/abs/2108.07594)  
[&nbsp; ] Improve: [Adaptive Reasonance](https://arxiv.org/pdf/1905.11437)  

## Quick Start

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

```
$ python main.py
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