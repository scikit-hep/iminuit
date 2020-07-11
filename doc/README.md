iminuit logo
============

The iminuit logo uses glyphs generated from the free Gentium Plus font.

How to make a release
=====================

- Merge develop into master
  - Option: do a `git rebase -i` to make history edits
- Create a branch on the master, called release_<version>, make release edits there
  - Increase version number in iminuit/version.py
  - Update doc/changelog.rst
  - Check that all wheels are build (Azure should trigger)
- Run `make integration` to do integration tests (if these fail, add tests to iminuit!)
- Merge release branch to master
- Create release on Github
  - This triggers an upload of the latest build artefacts to PyPI
  - Note: a prerelease is published on TestPyPI, a release on PyPI
  - Upload uses API tokens configured in PyPI, TestPyPI, and Github "Secrets"
- Conda-forge should pick this up automatically and generate conda packages
