import pytest
import re
import os

pj = os.path.join

nbformat = pytest.importorskip("nbformat")
preproc = pytest.importorskip("nbconvert.preprocessors")


class Processor(preproc.ExecutePreprocessor):
    "Executes timeit magic only once to run notebooks faster"

    def preprocess_cell(self, cell, resources, index):
        code = cell["source"]
        cell["source"] = re.sub("%?%timeit *", "", code)
        return super().preprocess_cell(cell, resources, index)


dir = os.path.dirname(__file__)
filenames = sorted([x for x in os.listdir(dir) if x.endswith("ipynb")])


@pytest.mark.parametrize("filename", filenames)
def test_notebook(filename):
    with open(pj(dir, filename)) as f:
        nb = nbformat.read(f, as_version=4)
        ep = Processor(timeout=1000)
        ep.preprocess(nb, {})
