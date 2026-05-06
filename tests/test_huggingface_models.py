from helpers import *
import pytest

def test_distilbert(data_dir):
    # lsshleifer/tiny-distilbert-base-cased
    check_model(data_dir / "pt3", {'Model': 'some',
                                    'Vector-matrix multiplications found': 17,
                                    'ops': [{'Output': (32, 2), 'Weights': (2, 2), 'Input': (32, 2)},
                                    {'Output': (32, 2), 'Weights': (2, 2), 'Input': (32, 2)},
                                    {'Output': (32, 2), 'Weights': (2, 2), 'Input': (32, 2)},
                                    {'Output': (2, 32, 32), 'Weights': (2, 1, 32), 'Input': (2, 32, 1)},
                                    {'Output': (2, 32, 1), 'Weights': (2, 32, 1), 'Input': (2, 32, 32)},
                                    {'Output': (32, 2), 'Weights': (2, 2), 'Input': (32, 2)},
                                    {'Output': (32, 4), 'Weights': (2, 4), 'Input': (32, 2)},
                                    {'Output': (32, 2), 'Weights': (4, 2), 'Input': (32, 4)},
                                    {'Output': (32, 2), 'Weights': (2, 2), 'Input': (32, 2)},
                                    {'Output': (32, 2), 'Weights': (2, 2), 'Input': (32, 2)},
                                    {'Output': (32, 2), 'Weights': (2, 2), 'Input': (32, 2)},
                                    {'Output': (2, 32, 32), 'Weights': (2, 1, 32), 'Input': (2, 32, 1)},
                                    {'Output': (2, 32, 1), 'Weights': (2, 32, 1), 'Input': (2, 32, 32)},
                                    {'Output': (32, 2), 'Weights': (2, 2), 'Input': (32, 2)},
                                    {'Output': (32, 4), 'Weights': (2, 4), 'Input': (32, 2)},
                                    {'Output': (32, 2), 'Weights': (4, 2), 'Input': (32, 4)},
                                    {'Output': (1, 2), 'Weights': (2, 2), 'Input': (1, 2)}]})
    
def test_bert_tiny(data_dir):
    # "prajjwal1/bert-tiny"
    check_model(data_dir / "pt4", {'Model': 'some',
                                    'Vector-matrix multiplications found': 17,
                                    'ops': [{'Output': (8, 128), 'Weights': (128, 128), 'Input': (8, 128)},
                                    {'Output': (8, 128), 'Weights': (128, 128), 'Input': (8, 128)},
                                    {'Output': (8, 128), 'Weights': (128, 128), 'Input': (8, 128)},
                                    {'Output': (2, 8, 8), 'Weights': (2, 64, 8), 'Input': (2, 8, 64)},
                                    {'Output': (2, 8, 64), 'Weights': (2, 8, 64), 'Input': (2, 8, 8)},
                                    {'Output': (8, 128), 'Weights': (128, 128), 'Input': (8, 128)},
                                    {'Output': (8, 512), 'Weights': (128, 512), 'Input': (8, 128)},
                                    {'Output': (8, 128), 'Weights': (512, 128), 'Input': (8, 512)},
                                    {'Output': (8, 128), 'Weights': (128, 128), 'Input': (8, 128)},
                                    {'Output': (8, 128), 'Weights': (128, 128), 'Input': (8, 128)},
                                    {'Output': (8, 128), 'Weights': (128, 128), 'Input': (8, 128)},
                                    {'Output': (2, 8, 8), 'Weights': (2, 64, 8), 'Input': (2, 8, 64)},
                                    {'Output': (2, 8, 64), 'Weights': (2, 8, 64), 'Input': (2, 8, 8)},
                                    {'Output': (8, 128), 'Weights': (128, 128), 'Input': (8, 128)},
                                    {'Output': (8, 512), 'Weights': (128, 512), 'Input': (8, 128)},
                                    {'Output': (8, 128), 'Weights': (512, 128), 'Input': (8, 512)},
                                    {'Output': (1, 128), 'Weights': (128, 128), 'Input': (1, 128)}]})