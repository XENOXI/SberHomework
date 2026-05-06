import re
import pytest
import torch.nn as nn
import tempfile
import torch
import subprocess, sys
from pathlib import Path

def parse_shape(s: str) -> tuple:
    return tuple(int(x) for x in s.split("x"))

def extract_block(log_text: str) -> str:
    splited = log_text.split("\n")

    data = []
    start = False
    lines_from_last = 0
    sep = "="*10
    for line in splited:
        
        if sep in line:
            start = True
            lines_from_last = 0
        elif start:
            lines_from_last += 1
            data.append(line)
    return data if lines_from_last == 0 else data[:-lines_from_last]


def parse_log(log_text: str) -> dict:
    block = extract_block(log_text)
    result = {}

    buff = block[0]
    result["Model"] = buff[buff.rfind("Model:"):].split()[-1]
    buff = block[1]
    result["Vector-matrix multiplications found"] = int(buff[buff.rfind("Vector-matrix multiplications found:"):].split()[-1])
    result["ops"] = []
    for ln in block[3:]:
        splited = ln.split()    
        result["ops"].append({
            "Output": parse_shape(splited[-1]),
            "Weights": parse_shape(splited[-2]),
            "Input": parse_shape(splited[-3]),
        })

    return result




def call_program(model_file: Path) -> tuple[str, str, str]:
    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir)
        name_of_model = model_file.stem

        result = subprocess.run(
            [
                sys.executable,
                "main.py",
                "--model", str(model_file),
                "--out", tempdir,
            ],
            capture_output=True,
            text=True,
        )

        hlo_p = path / f"{name_of_model}_hlo.txt"
        data_p = path / f"{name_of_model}_data.json"
        print(result.stdout)
        assert hlo_p.is_file() and data_p.is_file(), \
            f"Файлы не созданы. stderr:\n{result.stderr}"

        hlo = hlo_p.read_text()
        data = data_p.read_text()

    return hlo, data, result.stdout

def check_model(fold: str, log: dict):
    fold = Path(fold).resolve()
    hlo, data, gotted_log = call_program(fold / "some.py")
    with open(fold / "some_hlo.txt") as f:
        assert f.read() in hlo
    with open(fold / "some_data.json") as f:
        assert data == f.read()
    print(gotted_log)
    assert parse_log(gotted_log) == log, parse_log(gotted_log)

@pytest.fixture(scope="session")
def data_dir() -> Path:
    return Path(__file__).parent / "data"