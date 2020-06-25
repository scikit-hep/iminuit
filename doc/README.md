iminuit logo
============

The iminuit logo uses glyphs generated from the free Gentium Plus font.

How to make a release
=====================
- merge develop into master (possibly merge into branch first, to do history edits)
- create a branch on the master, called release_<version>, make release edits there
- increase version number in iminuit/version.py
- update doc/changelog.rst
- run `make integration` to do integration tests (if these fail, add tests to iminuit!)
- merge release branch to master
- create release on Github
  - triggers an upload of the latest build artefacts to PyPI
  - prerelease are published on TestPyPI
  - upload uses API tokens configured in PyPI, TestPyPI, and Github "Secrets"
- conda-forge should pick this up automatically and generate conda packages
