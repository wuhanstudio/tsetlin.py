# Tsetlin Machine

> The Iris Dataset

- Import / output model
- Print clause output & class sum
- Vanilla TM / coelesed weighted TM (same states with different weights)

- PYNQ / HLS (C++) Vitis AI / AXI Lite
- **Unsupervised TM** & Adaptive Reasonance (ICML Jan then Journal)

- Tracing / Debugging Strategy / ISTM August

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