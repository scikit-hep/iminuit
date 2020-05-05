import pytest
import glob
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import re


class Processor(ExecutePreprocessor):
    "Executes timeit magic only once to run notebooks faster"

    def preprocess_cell(self, cell, resources, index):
        code = cell["source"]
        cell["source"] = re.sub("%?%timeit *", "", code)
        return super().preprocess_cell(cell, resources, index)


@pytest.mark.parametrize("filename", sorted(glob.glob("tutorial/*.ipynb")))
def test_notebook(filename):
    with open(filename) as f:
        nb = nbformat.read(f, as_version=4)
        ep = Processor(timeout=1000)
        ep.preprocess(nb, {})
