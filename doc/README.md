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
- run `.ci/download_azure_artifacts.py` to download all wheels from the latest pipeline
- run `python -m twine upload dist/*` if everything looks ok
  (missing files can be uploaded later, but existing files cannot be overridden!)
- merge release branch to master
- create release on Github
- conda-forge should pick this up automatically and generate conda packages
