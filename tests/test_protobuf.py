import unittest

import tsetlin_pb2

class TestProtobuf(unittest.TestCase):
    def test_tsetlin_serialization(self):
        # Create a Tsetlin object and set its fields
        tsetlin = tsetlin_pb2.Tsetlin()

        tsetlin.n_class = 3
        tsetlin.n_feature = 32
        tsetlin.n_clause = 100
        tsetlin.n_state = 400

        tsetlin.model_type = tsetlin_pb2.ModelType.INFERENCE

        assert tsetlin.n_class == 3
        assert tsetlin.n_feature == 32
        assert tsetlin.n_clause == 100
        assert tsetlin.n_state == 400
        assert tsetlin.model_type == tsetlin_pb2.ModelType.INFERENCE
