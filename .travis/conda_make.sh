# From https://conda.io/docs/bdist_conda.html
# bdist_conda must be installed into a root conda environment,
# as it imports conda and conda_build. It is included as part of the conda build package.
conda install -n root conda-build Cython numpy
conda info
conda --version
conda build --version
make conda
#      source activate root;