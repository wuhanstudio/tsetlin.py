# Simulator (Online)

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


# Simulator (offline)

Save the raw data as `*.bin`:

```
$ python data.py
```

Upload the generated `*.bin` data to micropython:

```
$ mpremote fs cp main.bin :/main.bin
```
