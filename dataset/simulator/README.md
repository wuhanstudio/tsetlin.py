## NIML Data Simulator

Create the virtual environment:

```
$ uv venv --python 3.11
$ uv sync
```

Start the server:

```
# Linux
$ . .venv/bin/activate

# Windows
$ .venv\Scripts\activate

$ python -m flask --app server run --host=0.0.0.0
```

Start the client:

```
# Linux
$ . .venv/bin/activate

# Windows
$ .venv\Scripts\activate

$ python client.py
```
