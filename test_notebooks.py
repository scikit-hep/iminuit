"""Test if tutorial notebooks execute without error."""
import sys
from glob import glob
from time import time
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

STATUS_FAIL, STATUS_SUCCEED = 1, 0


def test_notebook(filename):
    t_start = time()
    with open(filename) as f:
        print("Executing notebook : {0}".format(filename))
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=1000, kernel_name='python')
        try:
            ep.preprocess(nb, {})
        except Exception as e:
            print("{0} [FAILED]\n{1}".format(filename, e))
            return STATUS_FAIL
        print("{0} [PASSED]".format(filename))
    t_run = time() - t_start
    print('Execution time for {}: {:.2f} sec'.format(filename, t_run))
    return STATUS_SUCCEED


def test_all_notebooks():
    filenames = sorted(glob("tutorial/*.ipynb"))
    broken_notebooks = [
        # See https://github.com/iminuit/iminuit/pull/245#issuecomment-402431753
        'tutorial/hard_core_tutorial.ipynb',
    ]
    status_total = STATUS_SUCCEED
    for filename in filenames:
        if filename in broken_notebooks:
            print('Skipping broken tutorial notebook: {}'.format(filename))
            continue
        status = test_notebook(filename)
        if status == STATUS_FAIL:
            status_total = STATUS_FAIL

    print('Total notebook execution status: {}'.format(status_total))
    return status_total


if __name__ == '__main__':
    print('Starting notebook tests')
    print('Python executable: {}'.format(sys.executable))
    print('Python version: {}'.format(sys.version))
    exit(test_all_notebooks())
