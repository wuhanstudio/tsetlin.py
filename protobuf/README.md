## Protobuf

Prerequisites:

``
$ sudo apt install protobuf-compiler

$ git submodule init
$ git submodule update

$ uv sync
$ source .venv/bin/activate
```

Python:

```
$ protoc --python_out=./ tsetlin.proto
```

MicroPython:

```
$ protoc --plugin=protoc-gen-custom=uprotobuf/scripts/uprotobuf_plugin.py --custom_out=. tsetlin.uproto
```

