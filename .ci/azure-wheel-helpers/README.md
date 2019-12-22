## Azure Wheel Helpers

This repository holds a collection of wheel helpers designed by the [Scikit-HEP][] project to build Python Wheels on [Azure DevOps][]. This is designed for packages that require building; if you have a pure-Python project, producing a universal wheel is trivial without this helper collection. This collection assumes some standard paths and procedures, though *some* of them can be customized. 

### Supported platforms and caveats

TLDR: Python 2.7, 3.6, 3.7, and 3.8  on all platforms, along with 3.5 on Linux.

| System | Arch | Python versions |
|---------|-----|------------------|
| SDist (all) | all |  any (non-binary distribution) |
| ManyLinux1 (custom GCC 8.3) | 64 & 32-bit | 2.7, 3.5, 3.6, 3.7, 3.8 |
| ManyLinux2010 | 64-bit | 2.7, 3.5, 3.6, 3.7, 3.8 |
| macOS 10.9+ | 64-bit | 2.7, 3.6, 3.7, 3.8 |
| Windows | 64 & 32-bit | 2.7, 3.6, 3.7, 3.8 |

* Linux: Python 3.4 is not supported because Numpy does not support it either.
* manylinux1: Optional support for GCC 9.1 using docker image; should work but can't be called directly other compiled extensions unless they do the same thing (think that's the main caveat). Supporting 32 bits because it's there for Numpy and PPA for now.
* manylinux2010: Requires pip 10+ and a version of Linux newer than 2010. This is very new technology. 64-bit only. Eventually this will become the preferred (and then only) way to produce Linux wheels.
* MacOS: Uses the dedicated 64 bit 10.9+ Python.org builds. We are not supporting 3.5 because those no longer provide binaries (could use 32+64 fat 10.6+ but really force to 10.9+, but will not be added unless there is a need for it).
* Windows: PyBind11 requires compilation with a newer copy of Visual Studio than Python 2.7's Visual Studio 2008; you need to have the [Visual Studio 2015 distributable][msvc2015] installed (the dll is included in 2017 and 2019, as well).

[msvc2017]: https://www.microsoft.com/en-us/download/details.aspx?id=48145

### Usage

> Azure does not recognize git submodules during the configure phase. Therefore, we are using git subtree instead.

This repository should reside in `/.ci` in your project. To add it:

```bash
git subtree add --prefix .ci/azure-wheel-helpers git@github.com:scikit-hep/azure-wheel-helpers.git master --squash
```

You should make a copy of the template pipeline and make local edits:

```bash
cp .ci/azure-wheel-helpers/azure-pipeline-build.yml .ci/azure-pipeline-build.yml
```

Make sure you enable this path in Azure as the pipeline. See [the post here][iscinumpy/wheels] for more details.

You must set the variables at the top of this file, and remove any configurations (like Windows) that you do not support:

```yaml
variables:
  package_name: my_package    # This is the output name, - is replaced by _
  many_linux_base: "quay.io/pypa/manylinux1_" # Could also be "skhep/manylinuxgcc-"
  dev_requirements_file: .ci/azure-wheel-helpers/empty-requirements.txt
  test_requirements_file: .ci/azure-wheel-helpers/empty-requirements.txt
```

You can adjust the rest of the template as needed. If you need a non-standard procedure, you can change the target of the `template` inputs to a local file.


### License

Copyright (c) 2019, Henry Schreiner.

Distributed under the 3-clause BSD license, see accompanying file LICENSE
or <https://github.com/scikit-hep/azure-wheel-helpers> for details.


[Scikit-HEP]:   http://scikit-hep.org
[Azure DevOps]: https://dev.azure.com
[iscinumpy/wheels]: https://iscinumpy.gitlab.io/post/azure-devops-python-wheels/
[msvc2017]: https://www.microsoft.com/en-us/download/details.aspx?id=48145

