#!/bin/bash
set -e -x

# Collect the pythons
pys=(/opt/python/*/bin)

# Print list of Python's available
echo "All Pythons: ${pys[@]}"

# Filter out Python 3.4
pys=(${pys[@]//*34*/})

# Filter out Python 3.8 on manylinux1
if [[ $PLAT =~ "manylinux1" ]]; then
    pys=(${pys[@]//*38*/})
fi

# Print list of Python's being used
echo "Using Pythons: ${pys[@]}"

# Compile wheels
for PYBIN in "${pys[@]}"; do
    "${PYBIN}/pip" install -r /io/.ci/requirements-build.txt
    "${PYBIN}/pip" wheel /io/ -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/$package_name-*.whl; do
    auditwheel repair --plat $PLAT "$whl" -w /io/wheelhouse/
done

# Install packages and test
for PYBIN in "${pys[@]}"; do
    "${PYBIN}/python" -m pip install $package_name --no-index -f /io/wheelhouse
    if [ -d "/io/tests" ]; then
        "${PYBIN}/pytest" /io/tests
    else
        "${PYBIN}/pytest" --pyargs $package_name
    fi
done
