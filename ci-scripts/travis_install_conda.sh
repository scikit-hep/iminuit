# Set up miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda --version
conda info -a

# From https://conda.io/docs/bdist_conda.html
# bdist_conda must be installed into a root conda environment,
# as it imports conda and conda_build. It is included as part of the conda build package.
conda build --version
conda install conda-build Cython numpy pytest matplotlib scipy ipython sphinx jupyter
