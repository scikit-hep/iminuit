import pytest
import re
import os
import shlex

pj = os.path.join

nbformat = pytest.importorskip("nbformat")
preproc = pytest.importorskip("nbconvert.preprocessors")


class Processor(preproc.ExecutePreprocessor):
    "Executes timeit magic only once to run notebooks faster"

    def preprocess_cell(self, cell, resources, index):
        code = cell["source"]

        # remove %timeit magic to execute cell only once
        # regex matches %timeit and %%timeit with cmdline arguments
        lines = code.split("\n")
        tmp = []
        for line in lines:
            if "%timeit" in line:
                items = shlex.split(line)
                istart = 0
                for i, x in enumerate(items):
                    if x.endswith("%timeit"):
                        istart = i
                    if x.startswith("-"):
                        istart = i + 2
                line = " ".join(items[istart:])
            tmp.append(line)
        cell["source"] = "\n".join(tmp)

        # execute cell
        cell, resources = super().preprocess_cell(cell, resources, index)

        # assert that there are no DeprecationWarning in output
        for output in cell.get("outputs", []):
            if output["output_type"] != "stream":
                continue
            text = output["text"]
            # temporary fix
            assert (
                "Creating an ndarray from ragged nested sequences" in text
                or "DeprecationWarning" not in text
            ), ("DeprecationWarning in cell [%i]" % cell["execution_count"])

        return cell, resources


dir = os.path.dirname(__file__)
filenames = sorted([x for x in os.listdir(dir) if x.endswith("ipynb")])


@pytest.mark.parametrize("filename", filenames)
def test_notebook(filename):
    with open(pj(dir, filename)) as f:
        nb = nbformat.read(f, as_version=4)
        ep = Processor(timeout=1000)
        ep.preprocess(nb, {"metadata": {"path": dir + "/../src"}})
