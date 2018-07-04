"""Test if tutorial notebooks execute without error."""
import sys
from glob import glob
from time import time
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError

def test_notebook(filename):
    t_start = time()
    with open(filename) as f:
        print("Executing notebook : {0}".format(filename))
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=1000, kernel_name='python3')
        try:
            ep.preprocess(nb, {'metadata': {'path': 'tutorial/'}})
        except CellExecutionError as e:
            print("{0} [FAILED]\n{1}".format(filename, e))
            exit(1)
        print("{0} [PASSED]".format(filename))
    t_run = time() - t_start
    print('Execution time for {}: {} sec'.format(filename, t_run))

def test_all_notebooks():
    filenames = sorted(glob("tutorial/*.ipynb"))
    for filename in filenames:
        test_notebook(filename)

if __name__ == '__main__':
    print('Starting notebook tests')
    print('Python executable: {}'.format(sys.executable))
    print('Python version: {}'.format(sys.version))
    test_all_notebooks()
