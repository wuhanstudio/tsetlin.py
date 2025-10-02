## NIML Data Simulator

Install python libraries:

```
$ uv venv --python 3.11
$ uv pip install git+https://github.com/nilmtk/nilmtk.git
$ uv pip install flask
$ uv pip install matplotlib
```

Start the server:

```
$ .venv\Scripts\activate
$ python -m flask --app server run --host=0.0.0.0
```

Start the client:

```
$ .venv\Scripts\activate
$ python client.py
```