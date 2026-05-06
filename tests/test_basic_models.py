from helpers import *
import pytest

def test_linear(data_dir):
    check_model(data_dir / "pt1", {"Model": "some", 
                               "Vector-matrix multiplications found": 1,  
                               "ops": [{"Input":  (1, 10), 
                                        "Output": (1, 20), 
                                        "Weights": (10, 20)}
                                ]})
    
def test_linear_relu(data_dir):
    check_model(data_dir / "pt2", {"Model": "some", 
                               "Vector-matrix multiplications found": 1,  
                               "ops": [{"Input":  (1, 20), 
                                        "Output": (1, 30), 
                                        "Weights": (20, 30)}
                                ]})
    
