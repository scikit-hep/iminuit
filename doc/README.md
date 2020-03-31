iminuit logo
============

The iminuit logo uses glyphs generated from the free Gentium Plus font.

How to make a release
=====================
- create a branch on the master, called release_<version>
- merge develop into this branch
- increase version number in iminuit/version.py
- update doc/changelog.rst
- check that new tutorials are list in the tutorials section of the docs
- run `make integration` to do integration tests (if these fail, add tests to iminuit!)
- deploy binaries to PyPI with Azure
- tag release on Github
